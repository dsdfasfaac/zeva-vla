"""Representation objectives with explicit positive and negative eligibility."""
import torch
from torch.nn import functional as F


def global_supervised_contrastive(first, second, task_ids, temperature=0.1):
    """Two augmented episode views; same-task episodes are additional positives.

    The two views must be computed without language. The diagonal is excluded;
    the matching augmented view remains a valid positive even in small batches.
    """
    vectors = F.normalize(torch.cat((first, second)), dim=-1)
    labels = task_ids.reshape(-1).repeat(2)
    diagonal = torch.eye(len(vectors), dtype=torch.bool, device=vectors.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = vectors @ vectors.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_prob = logits - denominator[:, None]
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return first.sum() * 0
    return -(log_prob.masked_fill(~positives, 0).sum(dim=1)[valid] / counts[valid]).mean()


def variance_covariance_loss(features):
    """Prevent unit-length tokens from collapsing to a constant direction.

    Unit-normalized vectors have variance roughly 1/D for an isotropic cloud;
    rescale by sqrt(D) before applying the per-coordinate standard-deviation
    floor. This avoids an impossible std=1 demand on unit-norm features.
    """
    if len(features) < 2:
        return features.sum() * 0
    scaled = features * features.shape[-1] ** 0.5
    centered = scaled - scaled.mean(dim=0)
    covariance = centered.T @ centered / (len(features) - 1)
    variance = F.relu(1 - (covariance.diagonal() + 1e-4).sqrt()).mean()
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance + off_diagonal.square().sum() / features.shape[-1]


def causal_effect_contrastive(signals, targets, temperature=0.1):
    """Match a transition to its detached observed effect among other effects.

    This provides effect-shuffle negatives but is NOT an action intervention
    or proof of causality. Those controls require separately rerunning the
    encoder with the executed-action input shuffled.
    """
    if len(signals) < 2:
        return signals.sum() * 0
    logits = F.normalize(signals, dim=-1) @ F.normalize(targets.detach(), dim=-1).T / temperature
    labels = torch.arange(len(signals), device=signals.device)
    return F.cross_entropy(logits, labels)

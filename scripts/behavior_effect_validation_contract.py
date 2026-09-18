"""Pure, cache-independent validation decision enumeration and fixed controls."""
from collections import defaultdict
import hashlib
import random


def enumerate_decisions(records, tasks):
    """Derive the H15 grid from raw episode lengths, never cached entries/errors."""
    result = []
    for index, record in enumerate(records):
        task = record["key"][1]
        length = int(record["length"])
        if task not in tasks or length <= 15:
            continue
        record_id = f"{record['key'][0]}:{task}:{record['episode_index']}"
        starts = list(range(0, length - 15, 15))
        for t, frame in enumerate(starts + [starts[-1] + 15]):
            result.append({"sample_id": f"{record_id}@{frame}", "record_id": record_id,
                           "record_index": index, "task": task, "frame": frame, "cache_index": t})
    ids = [row["sample_id"] for row in result]
    if not result or len(ids) != len(set(ids)) or {r["task"] for r in result} != set(tasks):
        raise ValueError("Raw validation must have unique decisions and cover every declared task.")
    return sorted(result, key=lambda row: row["sample_id"])


def within_task_permutation(decisions, seed=20260918):
    """Fixed label-free derangement: shuffled order then one-position rotation."""
    groups = defaultdict(list)
    for row in decisions:
        groups[row["task"]].append(row["sample_id"])
    output = {}
    for task, ids in sorted(groups.items()):
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("Each task requires at least two unique decisions for derangement.")
        ids.sort()
        random.Random(f"{seed}:{task}").shuffle(ids)
        output.update(zip(ids, ids[1:] + ids[:1]))
    return output


def decision_noise_seed(sample_id):
    # Stable across processes, machines and iteration order; not Python hash().
    return int.from_bytes(hashlib.sha256(f"zeva-validation5-v1:{sample_id}".encode()).digest()[:8],
                          "big") % (2**63 - 1)


def check_cache_coverage(decisions, cache):
    expected = defaultdict(list)
    for row in decisions:
        expected[row["record_id"]].append(row["frame"])
    if set(expected) != set(cache):
        raise ValueError("Validation cache episode coverage differs from raw validation records.")
    for record_id, frames in expected.items():
        entry = cache[record_id]
        observed = entry["frames"].tolist()
        if observed != sorted(frames):
            raise ValueError(f"Missing, extra, duplicated or reordered cache frames: {record_id}")
        if tuple(entry["phase"].shape) != (len(frames), 256) or tuple(entry["effect"].shape) != (len(frames), 256):
            raise ValueError(f"Bad recurrent cache shape: {record_id}")

#!/usr/bin/env python3
"""Canonical CLI for the ZeVA cross-attempt PIM training setting."""

import tyro

from scripts.train_robotwin_pim_policy import Args
from scripts.train_robotwin_pim_policy import main

if __name__ == "__main__":
    main(tyro.cli(Args))

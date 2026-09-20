#!/usr/bin/env python3
"""Canonical CLI for label-free cross-attempt PIM pairing artifacts."""

import tyro

from scripts.build_robotwin_pim_artifacts import Args
from scripts.build_robotwin_pim_artifacts import main

if __name__ == "__main__":
    main(tyro.cli(Args))

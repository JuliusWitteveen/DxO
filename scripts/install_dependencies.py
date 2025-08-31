#!/usr/bin/env python3
"""Utility script to install MAI-DxO dependencies."""

import subprocess
import sys
import pkg_resources

REQUIREMENTS = [
    "swarms",
    "loguru",
    "pydantic",
    "python-dotenv",
    "streamlit",
    "plotly",
    "numpy",
]


def main() -> None:
    for req_spec in REQUIREMENTS:
        try:
            pkg_resources.require(req_spec)
            print(f"{req_spec} is already installed")
        except (
            pkg_resources.DistributionNotFound,
            pkg_resources.VersionConflict,
        ):
            print(f"Installing {req_spec} ...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", req_spec]
                )
            except subprocess.CalledProcessError as exc:
                print(f"Failed to install {req_spec}: {exc}")
                print("Please install this package manually.")
    print("Dependency installation complete.")


if __name__ == "__main__":
    main()

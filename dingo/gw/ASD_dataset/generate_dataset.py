import os

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)

import numpy as np
import glob
import yaml
from os.path import join
import logging
import argparse
import textwrap
from typing import Dict, List
from functools import partial
from tqdm import tqdm

from .dataset_utils import download_and_estimate_PSDs, create_dataset_from_files, estimate_PSDs_from_gwfs


def generate_dataset(data_dir: str, settings: dict, verbose=False):
    run = settings["dataset_settings"]["observing_run"]
    if "channels" in settings["dataset_settings"].keys():
        estimate_PSDs_from_gwfs(data_dir, settings, verbose=verbose)
    else:
        for ifo in settings["dataset_settings"]["detectors"]:
            print(f"Downloading PSD data for observing run {run} and detector {ifo}")
            download_and_estimate_PSDs(
                data_dir, run, ifo, settings["dataset_settings"], verbose=verbose
            )

    create_dataset_from_files(data_dir, run, settings["dataset_settings"]["detectors"], settings["dataset_settings"])



def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate an ASD dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path where the PSD data is to be stored. Must contain a 'settings.yaml' file.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel parameterisation",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Visualize progress with bars",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Load settings
    with open(join(args.data_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    generate_dataset(
        args.data_dir,
        settings,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

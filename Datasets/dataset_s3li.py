from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
from contextlib import suppress
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import csv
import yaml
import cv2
import os

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile


class S3LI_dataset(DatasetVSLAMLab):
    """DLR S3LI Etna & Vulcano dataset helper for VSLAMLab benchmark."""

    def __init__(self, benchmark_path: str | Path) -> None:
        super().__init__("s3li", Path(benchmark_path))
    
        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            
        # Get download url
        self.url_download_sequences = cfg["url_download_sequences"]
        

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        zip_path = self.dataset_path / f"{sequence_name}.zip"

        if not zip_path.exists():
            url = self.url_download_sequences[sequence_name]
            print("Downloading sequence from:", url)
            downloadFile(url, str(self.dataset_path), sequence_name + ".zip")
        
        if sequence_path.exists():
            shutil.rmtree(sequence_path)

        decompressFile(str(zip_path), str(sequence_path), True)


    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        pass

    def create_imu_csv(self, sequence_name: str) -> None:
        pass
        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        pass

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        pass

    def remove_unused_files(self, sequence_name: str) -> None:
        pass

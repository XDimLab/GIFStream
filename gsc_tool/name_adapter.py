import glob
import os
from pathlib import Path
import subprocess
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import shutil

import tyro

from scene_info import DATASET_INFOS

@dataclass
class ConversionConfig:
    """Configuration for YUV conversion"""
    scene: str
    """Scene name (e.g., Bartender)"""
    
    base_dir: Optional[str] = None
    """Base directory path. If not provided, defaults to examples/data/GSC/{scene}"""

def main(config: ConversionConfig):
    # Process parameters
    SCENE = config.scene
    BASE_DIR = config.base_dir if config.base_dir else f"examples/data/GSC/{SCENE}"

    # Get all the png files in the base directory
    png_files = glob.glob(os.path.join(BASE_DIR, "png", "*.png"))
    
    # Rename the png files  
    for png_file in tqdm(png_files):
        # Get the frame number from the filename
        frame_number = int(png_file.split("/")[-1].split(".")[0][-3:])
        view_number = int(png_file.split("/")[-1].split(".")[0][1:3])
        os.makedirs(os.path.join(BASE_DIR, "png"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "png", f"cam{view_number:02d}"), exist_ok=True)
        shutil.move(png_file, os.path.join(BASE_DIR, "png", f"cam{view_number:02d}", f"{frame_number:05d}.png"))

if __name__ == "__main__":
    config = tyro.cli(ConversionConfig)
    main(config)
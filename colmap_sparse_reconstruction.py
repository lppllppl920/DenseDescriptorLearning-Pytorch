'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dense Descriptor Learning -- sparse reconstruction using COLMAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--colmap_exe_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True, help='root of data')
    parser.add_argument("--sequence_root", type=str, default=None, help='root of video sequence')
    parser.add_argument("--output_mode", type=str, default=None, help='mode of output')
    parser.add_argument("--camera_model", type=str, required=True, help='model of the camera to use')
    args = parser.parse_args()

    colmap_exe_path = args.colmap_exe_path

    if args.sequence_root is not None:
        sequence_root_list = [Path(args.sequence_root)]
    else:
        sequence_root_list = sorted(list(Path(args.data_root).rglob("_start*")))

    for sequence_root in sequence_root_list:
        print(f"Processing {str(sequence_root)}...")
        database_path = sequence_root / f"database_{args.camera_model}.db"
        image_root = sequence_root / "images"
        output_root = sequence_root / "colmap"

        if not output_root.exists():
            output_root.mkdir(parents=True)

        if args.output_mode == "continue" and (output_root / "0").exists():
            os.system(
                "{} mapper --database_path \"{}\" --input_path \"{}\" --image_path \"{}\" --output_path \"{}\"".format(
                    colmap_exe_path, str(database_path), str(output_root / "0"), str(image_root),
                    str(output_root / "0")))
        elif args.output_mode == "overwrite" or not (output_root / "0").exists():
            # if not (output_root / "0").exists():
            #     (output_root / "0").mkdir(parents=True)
            os.system(
                "{} mapper --database_path \"{}\" --image_path \"{}\" --output_path \"{}\"".format(
                    colmap_exe_path, str(database_path), str(image_root), str(output_root)))
        else:
            print(f"output mode {args.output_mode} not supported or if statements above not satisfied")
            continue

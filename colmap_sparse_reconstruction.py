import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dense Descriptor Learning -- sparse reconstruction using COLMAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--colmap_exe_path", type=str, required=True)
    parser.add_argument("--sequence_root", type=str, required=True, help='root of video sequence')
    parser.add_argument("--overwrite_reconstruction", action="store_true")
    args = parser.parse_args()

    colmap_exe_path = args.colmap_exe_path
    sequence_root = Path(args.sequence_root)
    database_path = sequence_root / "database.db"
    image_root = sequence_root / "images"
    output_root = sequence_root / "colmap"
    overwrite_reconstruction = args.overwrite_reconstruction

    if not overwrite_reconstruction:
        item = list(output_root.glob("*"))
        if output_root.exists() and len(item) > 0:
            print("ERROR: reconstruction exists already")
            exit()

    if not output_root.exists():
        output_root.mkdir(parents=True)

    os.system(
        "{} mapper --database_path \"{}\" --image_path \"{}\" --output_path \"{}\"".format(
            colmap_exe_path, database_path, image_root, str(output_root)))

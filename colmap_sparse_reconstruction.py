import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dense Descriptor Learning -- sparse reconstruction using COLMAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sequence_root", type=str, required=True)
    parser.add_argument("--colmap_exe_path", type=str, required=True)

    args = parser.parse_args()
    sequence_root = Path(args.sequence_root)
    colmap_exe_path = args.colmap_exe_path
    database_path = sequence_root / "database.db"
    images_path = sequence_root / "images"

    os.system(
        "{} mapper --database_path \"{}\" --image_path \"{}\" --output_path \"{}\"".format(
            colmap_exe_path, str(database_path), str(images_path),
            str(sequence_root / "results")))

import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dense Descriptor Learning -- sparse reconstruction using COLMAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--colmap_exe_path", type=str, required=True)
    parser.add_argument("--database_path", type=str, required=True, help='path of colmap database file')
    parser.add_argument("--sequence_root", type=str, required=True, help='root of video sequence')
    parser.add_argument("--output_root", type=str, required=True, help='root of output SfM results')

    args = parser.parse_args()
    colmap_exe_path = args.colmap_exe_path
    sequence_root = args.sequence_root
    database_path = args.database_path
    output_root = Path(args.output_root)
    if not output_root.exists():
        output_root.mkdir(parents=True)

    os.system(
        "{} mapper --database_path \"{}\" --image_path \"{}\" --output_path \"{}\"".format(
            colmap_exe_path, database_path, sequence_root, str(output_root)))

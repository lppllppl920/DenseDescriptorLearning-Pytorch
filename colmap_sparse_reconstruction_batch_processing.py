import os
from pathlib import Path
import shutil

if __name__ == "__main__":
    root = Path("D:/Data/WholeModelReconstruction_video")
    colmap_exe_path = "D:/Data/COLMAP/COLMAP-dev-windows/COLMAP.bat"
    python_path = "D:/Research/Projects/venv/Scripts/python.exe"
    sequence_list = list(root.rglob("_start*"))
    sequence_list.sort()

    python_script = "D:/Research/Projects/DenseDescriptorLearning-Pytorch/colmap_sparse_reconstruction.py"
    for sequence_root in sequence_list:
        print("Processing {}...".format(str(sequence_root)))
        image_root = sequence_root / "images"
        database_path = sequence_root / "database.db"
        output_root = sequence_root / "colmap"

        items = list(output_root.glob("*"))
        if len(items) > 0 or not database_path.exists():
            continue

        if not image_root.exists():
            ori_image_root = sequence_root / "ori_images"
            if not ori_image_root.exists():
                image_root.mkdir(parents=False)
                image_path_list = list(sequence_root.glob("0*.jpg"))
                for image_path in image_path_list:
                    shutil.move(src=str(image_path), dst=str(image_root / image_path.name))
            else:
                image_root.mkdir(parents=False)
                image_path_list = list(ori_image_root.glob("0*.jpg"))
                for image_path in image_path_list:
                    shutil.move(src=str(image_path), dst=str(image_root / image_path.name))

        os.system(
            "{} \"{}\" --colmap_exe_path \"{}\" --sequence_root \"{}\"".format(
                python_path,
                python_script,
                colmap_exe_path,
                str(sequence_root)
            ))

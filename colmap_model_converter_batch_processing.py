import os
from pathlib import Path

if __name__ == "__main__":
    root = Path("D:/Data/WholeModelReconstruction_video")
    sequence_list = list(root.rglob("_start*"))
    sequence_list.sort()

    colmap_exe_path = "D:/Data/COLMAP/COLMAP-dev-windows/COLMAP.bat"
    python_path = "D:/Research/Projects/venv_py36_x64/Scripts/python.exe"
    script_path = "D:/Research/Projects/DenseDescriptorLearning-Pytorch/colmap_model_converter.py"
    for sequence_root in sequence_list:
        os.system(
            "{} {} --colmap_exe_path \"{}\" --sequence_root \"{}\"".
                format(python_path,
                       script_path,
                       str(colmap_exe_path),
                       str(sequence_root)))

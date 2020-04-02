import os
from pathlib import Path

if __name__ == "__main__":
    root = Path("/home/xingtong/RemoteData/Sinus Project Data/xingtong/FullLengthEndoscopicVIdeoData")
    temp_root = Path("/home/xingtong/database")
    sequence_list = list(root.rglob("_start*"))
    sequence_list.sort()

    python_path = "/home/xingtong/Research/VirtualEnvironment/py36/bin/python3"
    script_path = "/home/xingtong/Research/DenseDescriptorLearning-Pytorch/colmap_database_creation.py"
    for sequence_root in sequence_list:
        temp_sequence_root = (temp_root / sequence_root.parents[0].name / sequence_root.name)
        if not temp_sequence_root.exists():
            temp_sequence_root.mkdir(parents=True)

        os.system(
            "{} {} --sequence_root \"{}\" --feature_match_path \"{}\" --output_root \"{}\" --overwrite_database".
                format(python_path,
                       script_path,
                       str(sequence_root),
                       str(sequence_root / "feature_matches_fm_only_spatial_grouping.hdf5"),
                       str(temp_sequence_root)))

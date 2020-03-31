import os
from pathlib import Path

if __name__ == "__main__":
    root = Path("")
    sequence_list = list(root.rglob("_start*"))

    for sequence_root in sequence_list:
        os.system(
            "python3.6 --sequence_root \"{}\" --feature_matching_path \"{}\" --output_root\"{}\"".format(sequence_root,
                                                                                                         sequence_root,
                                                                                                         sequence_root))

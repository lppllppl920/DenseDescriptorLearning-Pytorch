if __name__ == "__main__":
    import os
    from pathlib import Path

    colmap_exe_path = "D:/Data/COLMAP/COLMAP-dev-windows/COLMAP.bat"
    sequence_root = Path(
        "D:/Data/WholeModelReconstruction_video/bag_11/_start_000001_end_000901_segment_stride_1000_frame_stride_0002_segment_0000")
    image_root = sequence_root / "images"
    database_path = sequence_root / "database.db"
    result_root = sequence_root / "colmap"

    result_path_list = list(result_root.glob("*"))

    for result_path in result_path_list:
        os.system(
            "{} model_converter --input_path \"{}\" --output_path \"{}\" --output_type TXT".format(
                colmap_exe_path, str(result_path), str(result_path)))

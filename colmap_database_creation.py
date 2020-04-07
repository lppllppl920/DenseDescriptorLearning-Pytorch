'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import argparse
from pathlib import Path
import sys
import sqlite3
import numpy as np
import cv2
import h5py
import tqdm
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2 ** 31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert (len(matches.shape) == 2)
        assert (matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert (len(matches.shape) == 2)
        assert (matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
                                          array_to_blob(F), array_to_blob(E), array_to_blob(H)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dense Descriptor Learning -- COLMAP database building',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sequence_root", type=str, required=True, help='root of video sequence')
    parser.add_argument("--overwrite_database", action="store_true")

    args = parser.parse_args()
    sequence_root = Path(args.sequence_root)
    overwrite_database = args.overwrite_database
    feature_match_path = sequence_root / "feature_matches.hdf5"
    database_path = sequence_root / "database.db"

    if not overwrite_database:
        if database_path.exists():
            print("ERROR: database exists already")
            exit()

    if not feature_match_path.exists():
        print("ERROR: feature matches hdf5 file does not exist")
        exit()

    # Open the database.
    db = COLMAPDatabase.connect(str(database_path))

    # For convenience, try creating all the tables upfront.
    db.create_tables()

    images_root = sequence_root / "images"
    image_list = list(images_root.glob("0*.jpg"))
    image_list.sort()

    image = cv2.imread(str(image_list[0]))
    height, width, _ = image.shape

    # Create camera model -- PINHOLE CAMERA (fx fy cx cy))
    camera_intrinsics_path = sequence_root / "camera_intrinsics_per_view"
    with open(str(camera_intrinsics_path), "r") as f:
        temp = list()
        for i in range(4):
            temp.append(f.readline())
    model, intrinsics = 1, np.array((temp[0], temp[1], temp[2], temp[3]))
    camera_id = db.add_camera(model, width, height, intrinsics)

    # Create image ids
    image_id_list = list()
    for image_path in image_list:
        image_id_list.append(db.add_image(image_path.name, camera_id))

    # Create matches per image pair
    f_matches = h5py.File(str(feature_match_path), 'r')
    dataset_matches = f_matches['matches']
    start_index = 0

    tq = tqdm.tqdm(total=dataset_matches.shape[0])
    tq.set_description("Gathering keypoints")
    keypoints_dict = dict()
    # Keypoint gathering
    while start_index < dataset_matches.shape[0]:
        header = dataset_matches[start_index, :, 0]
        num_matches, id_1, id_2, _ = header
        tq.set_postfix(source_frame_index='{:d}'.format(id_1),
                       target_frame_index='{:d}'.format(id_2))
        id_1 += 1
        id_2 += 1
        id_1 = int(id_1)
        id_2 = int(id_2)
        pair_matches = dataset_matches[start_index + 1:start_index + 1 + num_matches, :, 0]
        pair_matches = pair_matches.astype(np.long)
        matches = np.concatenate([(pair_matches[:, 0] + pair_matches[:, 1] * width).reshape((-1, 1)),
                                  (pair_matches[:, 2] + pair_matches[:, 3] * width).reshape((-1, 1))], axis=1)

        if str(id_1) not in keypoints_dict:
            keypoints_dict[str(id_1)] = list(matches[:, 0])
        else:
            keypoints_dict[str(id_1)] += list(matches[:, 0])
            keypoints_dict[str(id_1)] = list(np.unique(keypoints_dict[str(id_1)]))

        if str(id_2) not in keypoints_dict:
            keypoints_dict[str(id_2)] = list(matches[:, 1])
        else:
            keypoints_dict[str(id_2)] += list(matches[:, 1])
            keypoints_dict[str(id_2)] = list(np.unique(keypoints_dict[str(id_2)]))
        tq.update(num_matches + 1)
        start_index += num_matches + 1

    tq.close()
    new_keypoints_dict = dict()
    # Keypoint indexing building
    # val -- 1D location of keypoint, key -- one-based frame index
    for key, value in keypoints_dict.items():
        value = np.unique(value)
        temp = dict()
        for idx, val in enumerate(value):
            temp[str(val)] = int(idx + 1)
        new_keypoints_dict[key] = temp

    tq = tqdm.tqdm(total=dataset_matches.shape[0])
    tq.set_description("Adding matches to database")
    start_index = 0
    while start_index < dataset_matches.shape[0]:
        header = dataset_matches[start_index, :, 0]
        num_matches, id_1, id_2, _ = header
        id_1 += 1
        id_2 += 1
        id_1 = int(id_1)
        id_2 = int(id_2)
        # new_keypoint_dict -- one-based frame index
        keypoint_dict_1 = new_keypoints_dict[str(id_1)]
        keypoint_dict_2 = new_keypoints_dict[str(id_2)]

        pair_matches = dataset_matches[start_index + 1:start_index + 1 + num_matches, :, 0]
        model, inliers = ransac((pair_matches[:, :2],
                                 pair_matches[:, 2:]),
                                FundamentalMatrixTransform, min_samples=8,
                                residual_threshold=10.0, max_trials=10)

        pair_matches = pair_matches.astype(np.long)
        pair_matches = np.concatenate([(pair_matches[:, 0] + pair_matches[:, 1] * width).reshape((-1, 1)),
                                       (pair_matches[:, 2] + pair_matches[:, 3] * width).reshape((-1, 1))], axis=1)

        idx_pair_matches = np.zeros_like(pair_matches).astype(np.int32)
        # one-based keypoint index per frame
        for i in range(pair_matches.shape[0]):
            idx_pair_matches[i, 0] = int(keypoint_dict_1[str(pair_matches[i, 0])])
            idx_pair_matches[i, 1] = int(keypoint_dict_2[str(pair_matches[i, 1])])
        db.add_matches(id_1, id_2, idx_pair_matches)

        # Fundamental matrix provided
        db.add_two_view_geometry(id_1, id_2, idx_pair_matches[inliers], F=model.params, config=3)

        tq.set_postfix(source_frame_index='{:d}'.format(id_1),
                       target_frame_index='{:d}'.format(id_2),
                       inliers_ratio='{:.3f}'.format(np.sum(inliers) / idx_pair_matches.shape[0]))
        tq.update(num_matches + 1)
        start_index += num_matches + 1

    tq.close()
    # Adding keypoints per image to database
    for image_id in image_id_list:
        if str(image_id) not in keypoints_dict:
            continue
        temp = np.asarray(keypoints_dict[str(image_id)])
        temp_1 = np.floor(temp.reshape((-1, 1)) / width)
        temp_2 = np.mod(temp.reshape((-1, 1)), width)
        db.add_keypoints(int(image_id), np.concatenate([temp_2, temp_1], axis=1).astype(np.int32))

    # Write to the database file
    db.commit()

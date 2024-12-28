import os
import pathlib
import pandas as pd

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

OUTPUT_DIR = pathlib.Path("./data/output")
INTERMEDIATE_DIR = pathlib.Path("./data/intermediate")

FPS = 15
NUMBER_OF_SUBJECTS = 63

validation_interval_in_minutes = 5 # in minutes
session_1_data = pd.read_hdf(INTERMEDIATE_DIR.joinpath(f"{FPS}_fps_data_{NUMBER_OF_SUBJECTS}_subjects.hdf5"),
                             key="subject_session_idx_0")
session_2_data = pd.read_hdf(INTERMEDIATE_DIR.joinpath(f"{FPS}_fps_data_{NUMBER_OF_SUBJECTS}_subjects.hdf5"),
                             key="subject_session_idx_1")

subject_ids = pd.DataFrame({"subject_ids": session_1_data.subject_id.unique()})
actual_subject_length = len(subject_ids)
assert NUMBER_OF_SUBJECTS == actual_subject_length, f"There was a problem in the aggregation step. " \
                                               f"This file should contain {NUMBER_OF_SUBJECTS} participants, " \
                                               f"but contains {actual_subject_length}"
print(f"The loaded data contains {actual_subject_length} subjects")

output_path_data = pathlib.Path(OUTPUT_DIR.joinpath(f"{FPS}_fps-{NUMBER_OF_SUBJECTS}_subjects-metric_learning_movement.hdf5"))
if output_path_data.exists():
    os.remove(output_path_data)


OFFSET_FRAMES = FPS * 60 * 1
CUTOFF_FRAMES = FPS * 60 * 1

num_of_frames_in_validation_data = FPS * 60 * validation_interval_in_minutes

subject_ids.to_hdf(output_path_data, key="subject_ids", mode="a", index=False, dropna=True, append=True)


for session_idx, data in enumerate([session_1_data, session_2_data]):
    for subject_id, features in data.groupby("subject_id"):

        features["session_idx"] = session_idx
        features["session_idx"] = features["session_idx"].astype("uint8")
        key = f"{subject_id}"
        features[OFFSET_FRAMES:-CUTOFF_FRAMES].to_hdf(output_path_data, key=key, mode="a", index=False, dropna=True, append=True, min_itemsize=11)

print(f"finished, saved to {output_path_data}")

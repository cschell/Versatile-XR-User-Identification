import numpy as np
import pandas as pd
import time

import pytest

from src.datamodules.datasets.window_maker import WindowMaker

WINDOW_SIZE = 30
FRAME_STEP_SIZE = 3


@pytest.mark.parametrize("is_acceleration", [True, False])
def test_window_maker(is_acceleration):
    # change_idxs = np.array([0, 460, 830])
    change_idxs = np.array([0, 46549, 90811, 129064, 186281, 230401, 291100, 335362, 393619, 448740, 502577, 535481, 585810, 647580, 703736, 762869, 793397, 854096, 916977, 968993, 1011067, 1048637, 1105854, 1149693, 1193813, 1231668, 1269868, 1309052, 1336219, 1375403, 1413656, 1462436, 1495673, 1523414])

    num_frames = 1_555_487
    window_size = 30
    frame_step_size = 3

    wm = WindowMaker(num_frames, window_size=window_size, frame_step_size=frame_step_size, change_idxs=change_idxs, is_acceleration=is_acceleration)

    window_span = (window_size + is_acceleration) * frame_step_size
    expected_num_windows = num_frames - window_span * len(change_idxs)

    assert wm.num_windows == expected_num_windows
    assert wm.windowed_ids.shape == (expected_num_windows, window_size)

    assert np.all(~np.isin(change_idxs, wm.windowed_ids)), "change_idxs must not appear in windows!"

    for step in range(1, frame_step_size):
        assert np.all(~np.isin(change_idxs + step, wm.windowed_ids)), f"frame_step_size is {frame_step_size}, so <change_idx> + {step} must not appear in windows!"



@pytest.mark.parametrize("is_acceleration", [True, False])
def test_window_maker_for_equal_sized_frames(is_acceleration):
    window_size = 43
    change_idxs = np.arange(0, 110655, 45)

    num_frames = 110655 + 1 # 10 * 45
    frame_step_size = 1

    wm = WindowMaker(num_frames, window_size=window_size, frame_step_size=frame_step_size, change_idxs=change_idxs, is_acceleration=is_acceleration)

    window_span = (window_size + is_acceleration) * frame_step_size
    expected_num_windows = num_frames - window_span * len(change_idxs)

    assert wm.num_windows == expected_num_windows
    assert wm.windowed_ids.shape == (expected_num_windows, window_size)

    assert np.all(~np.isin(change_idxs, wm.windowed_ids)), "change_idxs must not appear in windows!"

    for step in range(1, frame_step_size):
        assert np.all(~np.isin(change_idxs + step, wm.windowed_ids)), f"frame_step_size is {frame_step_size}, so <change_idx> + {step} must not appear in windows!"

    assert wm.windowed_ids.size > 0
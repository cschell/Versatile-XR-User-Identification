from unittest import TestCase
from scipy.spatial.transform import Rotation
from src.utils.rotation_helper import process_rotation_to_6d_representation, rotation_to_6d
from numpy.testing import assert_array_almost_equal
import numpy as np
import pandas as pd


class TestRotationHelper(TestCase):
    def setUp(self) -> None:
        self.test_data = pd.read_csv("test_data_for_6d_processing.csv")

        left_rot_6d = rotation_to_6d(Rotation.from_quat([-0.0027195316605099057, 0.008434854275899755,
                                                         -0.007846107966320418, -0.9999299455315221])).reshape(-1, 6)
        right_rot_6d = rotation_to_6d(Rotation.from_quat([-0.0016175876253099425, 0.008286868355642315,
                                                          -0.012365254230257833, -0.999887899572223])).reshape(-1, 6)
        head_rot_6d = rotation_to_6d(Rotation.from_quat([-0.0022080093237697098, 0.00015371640185394018,
                                                         -0.0005015485332177798, -0.9999974247542655])).reshape(-1, 6)
        self.third_row_expected = [
            # left_hand_pos x/y/z
            0.4585654997516686, 0.16522626074990399, -0.011985439158548772,
            # right_hand_pos x/y/z
            0.765527892328997, -0.047708664144067825, 0.23401969826254998,
            # left_hand_rot 1-6d
            # -0.0027195316605099057, 0.008434854275899755, -0.007846107966320418, -0.9999299455315221,
            left_rot_6d[0, 0], left_rot_6d[0, 1], left_rot_6d[0, 2],
            left_rot_6d[0, 3], left_rot_6d[0, 4], left_rot_6d[0, 5],
            # right_hand_rot 1-6d
            # -0.0016175876253099425, 0.008286868355642315, -0.012365254230257833, -0.999887899572223,
            right_rot_6d[0, 0], right_rot_6d[0, 1], right_rot_6d[0, 2],
            right_rot_6d[0, 3], right_rot_6d[0, 4], right_rot_6d[0, 5],
            # head_rot 1-6d
            # -0.0022080093237697098, 0.00015371640185394018, -0.0005015485332177798, -0.9999974247542655
            head_rot_6d[0, 0], head_rot_6d[0, 1], head_rot_6d[0, 2],
            head_rot_6d[0, 3], head_rot_6d[0, 4], head_rot_6d[0, 5],
        ]

    def test_rotation_to_6d(self):
        assert_array_almost_equal(np.array([[1, 0], [0, 1], [0, 0]]),
                                  rotation_to_6d(Rotation.from_euler("xyz", [0, 0, 0], degrees=True)))
        assert_array_almost_equal(np.array([[1, 0], [0, 0], [0, 1]]),
                                  rotation_to_6d(Rotation.from_euler("xyz", [90, 0, 0], degrees=True)))
        assert_array_almost_equal(np.array([[0, 0], [0, 1], [-1, 0]]),
                                  rotation_to_6d(Rotation.from_euler("xyz", [0, 90, 0], degrees=True)))
        assert_array_almost_equal(np.array([[0, -1], [1, 0], [0, 0]]),
                                  rotation_to_6d(Rotation.from_euler("xyz", [0, 0, 90], degrees=True)))

    def test_rotation_to_6d_given_multiple_rotations(self):
        expected = np.array([
            [[1, 0], [0, 1], [0, 0]],
            [[1, 0], [0, 0], [0, 1]],
            [[0, 0], [0, 1], [-1, 0]],
            [[0, -1], [1, 0], [0, 0]]
        ])

        actual = rotation_to_6d(Rotation.from_euler("xyz", np.array([[0, 0, 0], [90, 0, 0], [0, 90, 0], [0, 0, 90]]),
                                                    degrees=True))
        assert_array_almost_equal(expected, actual)

    def test_process_given_nan_values_then_keep_nan(self):
        two_rows_with_nan_values = self.test_data.iloc[:2]
        nan_row = [np.nan for _ in range(18)]
        expected_values = np.array([nan_row, nan_row])

        result = process_rotation_to_6d_representation(two_rows_with_nan_values)

        assert_array_almost_equal(expected_values, result.to_numpy())

    def test_process_given_single_row_then_remove_quat_and_add_6d_representation(self):
        expected_values = np.array([self.third_row_expected])

        result = process_rotation_to_6d_representation(self.test_data.iloc[2:3]).to_numpy()

        # left and right hand pos and left hand rot
        assert_array_almost_equal(expected_values[0, :12], result[0, :12])
        # right hand and head rot
        assert_array_almost_equal(expected_values[0, 12:24], result[0, 12:24])

    def test_process_given_multiple_rows_some_with_nan_values_then_parse_only_valid_ones(self):
        nan_row = [np.nan for _ in range(24)]
        expected_values = np.array([nan_row, nan_row, self.third_row_expected])

        result = process_rotation_to_6d_representation(self.test_data.iloc[:3]).to_numpy()

        # left and right hand pos and left hand rot
        assert_array_almost_equal(expected_values[:, :12], result[:, :12])
        # right hand and head rot
        assert_array_almost_equal(expected_values[:, 12:24], result[:, 12:24])

    def test_process_whole_file_without_error(self):
        result = process_rotation_to_6d_representation(self.test_data)

        self.assertEqual((20, 24), result.shape)
        self.assertEqual((2, 24), result[result.isna().any(axis=1)].shape)
        self.assertEqual((18, 24), result[result.notna().all(axis=1)].shape)

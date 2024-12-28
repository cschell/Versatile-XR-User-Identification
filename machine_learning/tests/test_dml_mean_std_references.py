import unittest
import pandas as pd
import torch
from analysis.dml_paper.computations.metrics_computation_helper import compute_mean_std_reference_and_query_data


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        test_file_name = "/storage/dml_paper/run_archive/1xnyyf13-lyric-planet-1/test_data_cache.feather"
        try:
            self.test_data = pd.read_feather(test_file_name)
        except FileNotFoundError:
            self.fail("Choose a valid test_file")

    def test_given_no_std_include_then_return_references_grouped_by_subject(self):
        r_e, r_y, q_e, q_y = compute_mean_std_reference_and_query_data(self.test_data, 150)

        self.assertEqual((27, 192), r_e.shape)
        self.assertEqual((27, ), r_y.shape)
        self.assertEqual(torch.Tensor, type(r_e))
        self.assertEqual(torch.Tensor, type(r_y))

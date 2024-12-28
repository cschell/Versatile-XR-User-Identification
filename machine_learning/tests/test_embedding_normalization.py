from unittest import TestCase
import torch
from src.utils.normalization_helper import normalize_embedding


class TestEmbeddingNormalization(TestCase):
    def setUp(self) -> None:
        self.test_data = torch.load('saved_embedded_tensor.pt')

    def test_given_non_normalized_tensor_then_normalize(self):
        all_ones = torch.ones((64,)).cuda()

        data_normalized = normalize_embedding(self.test_data)

        self.assert_normalized_afterwards(all_ones, data_normalized, self.test_data)

    def assert_normalized_afterwards(self, all_ones, data_normalized, data_before):
        actual_values = torch.linalg.norm(data_normalized, dim=0, ord=2)
        self.assertTrue(torch.allclose(all_ones, actual_values))
        norm_values_before = torch.linalg.norm(data_before, dim=0, ord=2)
        self.assertFalse(torch.allclose(all_ones, norm_values_before))

    def test_given_transposed_data_then_normalize(self):
        all_ones = torch.ones((400, )).cuda()
        test_data = torch.t(self.test_data)

        data_normalized = normalize_embedding(test_data)

        self.assert_normalized_afterwards(all_ones, data_normalized, test_data)


    def test_given_normalized_data_then_keep_it_normalized(self):
        all_ones = torch.ones((64, )).cuda()

        data_two_times_normalized = normalize_embedding(normalize_embedding(self.test_data))

        self.assert_normalized_afterwards(all_ones, data_two_times_normalized, self.test_data)

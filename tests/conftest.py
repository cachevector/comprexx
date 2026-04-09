"""Shared pytest fixtures for Comprexx tests."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.fixtures.models import tiny_cnn, tiny_rnn, tiny_transformer


@pytest.fixture
def cnn_model():
    return tiny_cnn()


@pytest.fixture
def transformer_model():
    return tiny_transformer()


@pytest.fixture
def rnn_model():
    return tiny_rnn()


@pytest.fixture
def dummy_calibration_loader():
    """DataLoader of random tensors for calibration."""
    def _make(input_shape=(64, 3, 32, 32), num_samples=64, batch_size=8):
        data = torch.randn(num_samples, *input_shape[1:])
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=batch_size)
    return _make

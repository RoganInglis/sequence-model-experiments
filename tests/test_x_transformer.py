import pytest
import torch

from src.models.components.x_transformer import XTransformer


device = 'cpu'


def test_x_transformer():
    vocab_size = 100
    max_length = 128
    batch_size = 2
    model = XTransformer(
        vocab_size=vocab_size,
        max_length=max_length
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)
    attention_mask = torch.ones(batch_size, max_length).to(device)

    y = model(input_ids, attention_mask)
    assert y.shape == (batch_size, max_length, vocab_size)
    assert y.dtype == torch.float32

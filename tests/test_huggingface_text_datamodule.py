import pytest
import torch

from src.data.huggingface_text_datamodule import HFTextDataModule


def test_huggingface_text_datamodule():
    max_length = 2048
    batch_size = 32
    dm = HFTextDataModule(
        dataset_name='roneneldan/TinyStories',
        tokenizer_name='openai-community/gpt2',
        max_length=max_length,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 2_163_699

    batch = next(iter(dm.train_dataloader()))
    assert 'input_ids' in batch
    assert 'attention_mask' in batch

    assert batch['input_ids'].shape == (batch_size, max_length)
    assert batch['attention_mask'].shape == (batch_size, max_length)
    assert batch['input_ids'].dtype == torch.int64
    assert batch['attention_mask'].dtype == torch.int64

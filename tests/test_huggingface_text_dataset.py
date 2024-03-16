import pytest

from src.data.components.huggingface_text_dataset import HFTextDataset


def test_huggingface_text_dataset():
    max_length = 2048
    dataset = HFTextDataset(
        dataset_name='roneneldan/TinyStories',
        tokenizer_name='openai-community/gpt2',
        max_length=max_length,
        split='train'
    )
    item = dataset[0]
    print(item)
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert item['input_ids'].shape[0] == max_length
    assert item['attention_mask'].shape[0] == max_length

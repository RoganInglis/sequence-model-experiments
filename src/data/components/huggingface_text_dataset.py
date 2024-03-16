from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class HFTextDataset(Dataset):
    def __init__(self, dataset_name,  tokenizer_name, max_length: int = 2048, dataset_config_name: Optional[str] = None, split: Optional[str] = None):
        self.dataset = load_dataset(dataset_name, dataset_config_name)
        if split is not None:
            self.dataset = self.dataset[split]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']

        # Tokenize the text
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length',
                                return_tensors='pt')

        # Extract input_ids and attention_mask
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze().to(torch.bool)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

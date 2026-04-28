import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
INLINE_PUNCTUATION = {"，", "、", "；", "？", "！"}
ENDING_PUNCTUATION = {"。", "？", "！"}


def _is_qiyan_jueju(paragraphs):
    if len(paragraphs) != 2:
        return False

    for line in paragraphs:
        content = line.strip()
        if len(content) != 16:
            return False
        if content[7] not in INLINE_PUNCTUATION:
            return False
        if content[15] not in ENDING_PUNCTUATION:
            return False
    return True


def load_qiyan_jueju_corpus(data_root):
    data_dir = Path(data_root)
    json_files = sorted(data_dir.glob("poet.song.4*.json"))
    if not json_files:
        raise FileNotFoundError(f"No song poetry json files found under {data_dir}")

    poems = []
    for json_file in json_files:
        records = json.loads(json_file.read_text(encoding="utf-8"))
        for item in records:
            paragraphs = item.get("paragraphs", [])
            if _is_qiyan_jueju(paragraphs):
                poems.append("".join(paragraphs))

    if not poems:
        raise RuntimeError("No seven-character quatrains were extracted from the dataset.")
    return poems


class PoetryVocabulary:
    def __init__(self, poems, min_freq=2):
        counter = Counter("".join(poems))
        charset = sorted(char for char, frequency in counter.items() if frequency >= min_freq)
        self.itos = list(SPECIAL_TOKENS) + charset
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

    def __len__(self):
        return len(self.itos)

    def encode_text(self, text):
        return [self.stoi.get(char, self.unk_id) for char in text]

    def decode_ids(self, token_ids):
        special_tokens = set(SPECIAL_TOKENS)
        chars = []
        for token_id in token_ids:
            token = self.itos[token_id]
            if token not in special_tokens:
                chars.append(token)
        return "".join(chars)


class PoetryDataset(Dataset):
    def __init__(self, poems, vocabulary):
        self.poems = poems
        self.vocabulary = vocabulary
        self.samples = []
        for poem in poems:
            token_ids = [vocabulary.bos_id]
            token_ids.extend(vocabulary.encode_text(poem))
            token_ids.append(vocabulary.eos_id)
            self.samples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence = self.samples[index]
        return sequence[:-1], sequence[1:]


def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

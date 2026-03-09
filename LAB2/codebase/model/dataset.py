from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class NERDatasetProcessor:
    def __init__(self, data_source: Path | pd.DataFrame, split: str):
        assert split in {"train", "test"}, "split must be 'train' or 'test'"
        self.data_source = data_source
        self.df = None
        self.split = split

    def load_data(self):
        """Load CSV dataset"""
        if isinstance(self.data_source, pd.DataFrame):
            self.df = self.data_source
        else:
            self.df = pd.read_csv(self.data_source)

    def clean_data(self):
        """Common processing + split-specific handling"""

        if "Unnamed: 0" in self.df.columns:
            self.df = self.df.drop(columns=["Unnamed: 0"])

        self.df["Sentence_id"] = self.df["Sentence_id"].ffill()

        if self.split == "train":
            self.df = self.df.dropna(subset=["Word"])

        elif self.split == "test":
            self.df["Word"] = self.df["Word"].fillna("[UNK]")

        self.df = self.df.reset_index(drop=True)

    def get_sentences(self):
        """Convert dataframe into sentences + labels"""

        sentences = []
        labels = []

        for _, group in self.df.groupby("Sentence_id"):
            words = group["Word"].tolist()

            if self.split == "train":
                tags = group["Tag"].tolist()
            else:
                tags = [0] * len(words)  # placeholder

            sentences.append(words)
            labels.append(tags)

        return sentences, labels
    

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def dataset_stats(sentences):
    lengths = [len(s) for s in sentences]

    print("Total sentences:", len(sentences))
    print("Average length:", sum(lengths)/len(lengths))
    print("Max length:", max(lengths))

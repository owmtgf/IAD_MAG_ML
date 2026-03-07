from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


ID2LABEL = {
    0:"O",
    1:"B-per",
    2:"I-per",
    3:"B-gpe",
    4:"I-gpe",
    5:"B-eve",
    6:"I-eve",
    7:"B-geo",
    8:"I-geo",
    9:"B-nat",
    10:"I-nat",
    11:"B-art",
    12:"I-art",
    13:"B-tim",
    14:"I-tim",
    15:"B-org",
    16:"I-org"
}

LABEL2ID = {v:k for k,v in ID2LABEL.items()}


class NERDatasetProcessor:

    def __init__(self, file_path: Path, split: str):
        assert split in {"train", "test"}, "split must be 'train' or 'test'"
        self.file_path = file_path
        self.df = None
        self.split = split

    def load_data(self):
        """Load CSV dataset"""
        self.df = pd.read_csv(self.file_path)

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
    

def dataset_stats(sentences):

    lengths = [len(s) for s in sentences]

    print("Total sentences:", len(sentences))
    print("Average length:", sum(lengths)/len(lengths))
    print("Max length:", max(lengths))


class NERKnowledgeBase:

    def __init__(self):
        self.categories = defaultdict(set)
        self.entity_texts = defaultdict(list)
        self.entity_meaning = dict()

    def add_entity(self, entity: str, category: str, texts=None, meaning=None):
        self.categories[category].add(entity)
        if texts:
            self.entity_texts[entity].extend(texts)
        if meaning:
            self.entity_meaning[entity] = meaning

    def delete_entity(self, entity: str):
        for cat, ents in self.categories.items():
            if entity in ents:
                ents.remove(entity)
        if entity in self.entity_texts:
            del self.entity_texts[entity]
        if entity in self.entity_meaning:
            del self.entity_meaning[entity]

    def add_words_from_text(self, text: str, category: str):
        words = text.split()
        for w in words:
            self.add_entity(w, category, texts=[text])

    def get_entity_info(self, entity: str):
        return {
            "categories": [cat for cat, ents in self.categories.items() if entity in ents],
            "texts": self.entity_texts.get(entity, []),
            "meaning": self.entity_meaning.get(entity, "")
        }

    def add_category(self, category: str, entities=None):
        if entities:
            self.categories[category].update(entities)

    def plot_wordcloud(self, category: str):
        if category not in self.categories:
            print(f"No such category: {category}")
            return
        
        words = " ".join(self.categories[category])
        wc = WordCloud(width=800, height=400, background_color="white").generate(words)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for category: {category}")
        plt.show()

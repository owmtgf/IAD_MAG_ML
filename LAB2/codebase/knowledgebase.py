import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from IPython.display import display, HTML
from tqdm import tqdm
from wordcloud import WordCloud

from .globals import ID2LABEL


class NERKnowledgeBase:
    def __init__(self):
        self.categories = defaultdict(set)
        self.entity_texts = defaultdict(list)
        self.entity_meaning = dict()

    def add_entity(self, entity: str, category: str, texts: list[str] = None, meaning: str = None):
        """
        Add a single entity to a category.
        Automatically avoids duplicates.
        """
        self.categories[category].add(entity)
        if texts:
            self.entity_texts[entity].extend(texts)
        if meaning:
            self.entity_meaning[entity] = meaning

    def delete_entity(self, entity: str):
        """Delete entity from all categories and texts/meanings"""
        for _, ents in self.categories.items():
            ents.discard(entity)
        self.entity_texts.pop(entity, None)
        self.entity_meaning.pop(entity, None)

    def add_category(self, category: str, entities: list[str] = None, reassign: bool = False):
        """
        Add a new category. Optionally assign entities.
        If reassign=True, entities will be removed from other categories automatically.
        """
        if entities:
            self.categories[category].update(entities)
            if reassign:
                for other_cat, ents in self.categories.items():
                    if other_cat == category:
                        continue
                    ents.difference_update(entities)

    def add_words_from_text(self, text: str, category: str):
        """
        Add words from raw text as entities.
        Uses simple filter to remove punctuation / stopwords (words with non-alpha chars).
        """
        for w in text.split():
            w_clean = w.strip().strip(".,!?;:\"()[]{}")
            if w_clean.isalpha():
                self.add_entity(w_clean, category, texts=[text])

    def add_entities_from_model(self, sentences: list[list[str]], word_tags: list[list[int]]):
        """
        Fill KB using NER model predictions.
        sentences: List of token lists
        word_tags: List of tag ids corresponding to sentences
        """
        for sent, tags in tqdm(zip(sentences, word_tags), desc="Populating KB from model"):
            entities = self.extract_entities(sent, tags)
            text = " ".join(sent)
            for entity, category in entities:
                self.add_entity(entity, category, texts=[text])

    def get_entity_info(self, entity: str):
        """Return category/categories, example texts, and meaning for a given entity"""
        return {
            "categories": [cat for cat, ents in self.categories.items() if entity in ents],
            "texts": self.entity_texts.get(entity, []),
            "meaning": self.entity_meaning.get(entity, "")
        }

    @staticmethod
    def extract_entities(words: list[str], tags: list[int]):
        """Convert list of words and predicted tag ids into entity spans"""
        entities = []
        current_entity = []
        current_type = None

        for word, tag_id in zip(words, tags):
            tag = ID2LABEL[tag_id]
            if tag == "O":
                if current_entity:
                    entities.append((" ".join(current_entity), current_type))
                    current_entity = []
                    current_type = None
                continue

            prefix, entity_type = tag.split("-")
            if prefix == "B":
                if current_entity:
                    entities.append((" ".join(current_entity), current_type))
                current_entity = [word]
                current_type = entity_type
            elif prefix == "I":
                current_entity.append(word)

        if current_entity:
            entities.append((" ".join(current_entity), current_type))

        return entities

    def plot_wordcloud(self, category: str):
        if category not in self.categories:
            print(f"No such category: {category}")
            return
        words = " ".join(self.categories[category])
        wc = WordCloud(width=800, height=400, background_color="white").generate(words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for category: {category}")
        plt.show()

    def highlight_sentence(self, words: list[str], tags: list[int], id2label: dict = None):
        """
        Highlight words in a sentence according to their NER category.

        Args:
            words: list of words in the sentence
            tags: list of predicted tag ids (aligned with words)
            id2label: optional mapping from tag id -> label string (default uses globals.ID2LABEL)
        """
        if id2label is None:
            from .globals import ID2LABEL
            id2label = ID2LABEL

        palette = list(mcolors.TABLEAU_COLORS.values())

        categories = sorted({id2label[t].split('-')[-1] for t in tags if id2label[t] != "O"})
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

        html_tokens = []
        for word, tag_id in zip(words, tags):
            label = id2label[tag_id]
            if label == "O":
                html_tokens.append(word)
            else:
                cat = label.split('-')[-1]
                color = color_map.get(cat, "#FFD700")
                html_tokens.append(
                    f'<span style="background-color:{color};padding:2px;margin:1px;border-radius:3px">{word}</span>'
                )

        html_str = " ".join(html_tokens)
        display(HTML(html_str))

    def save(self, path: str | Path):
        """Save KB to JSON"""
        data = {
            "categories": {k: list(v) for k, v in self.categories.items()},
            "entity_texts": dict(self.entity_texts),
            "entity_meaning": self.entity_meaning
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path):
        """Load KB from JSON"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kb = cls()
        for cat, entities in data["categories"].items():
            kb.categories[cat] = set(entities)
        kb.entity_texts = defaultdict(list, data["entity_texts"])
        kb.entity_meaning = data["entity_meaning"]
        return kb

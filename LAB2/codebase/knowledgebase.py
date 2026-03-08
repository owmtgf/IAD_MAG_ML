import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud

from .globals import ID2LABEL


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
        for _, ents in self.categories.items():
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
    
    def save(self, path: str | Path):
        data = {
            "categories": {k: list(v) for k, v in self.categories.items()},
            "entity_texts": dict(self.entity_texts),
            "entity_meaning": self.entity_meaning
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path):

        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        kb = cls()

        for category, entities in data["categories"].items():
            kb.categories[category] = set(entities)

        kb.entity_texts = defaultdict(list, data["entity_texts"])
        kb.entity_meaning = data["entity_meaning"]

        return kb


def populate_knowledge_base(knowledgebase, train_sentences, train_labels):
    if not knowledgebase:
        knowledgebase = NERKnowledgeBase()
    for words, tags in tqdm(zip(train_sentences, train_labels), desc="Populating knowledge base"):
        entities = extract_entities(words, tags)
        text = " ".join(words)

        for entity, category in entities:
            knowledgebase.add_entity(entity, category, texts=[text])
            
    return knowledgebase


def extract_entities(words, tags):

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

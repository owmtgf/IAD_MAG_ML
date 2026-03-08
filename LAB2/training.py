from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split

from dataset import (
    NERDatasetProcessor,
    NERKnowledgeBase,
    NERDataset,
    extract_entities,
    dataset_stats,
    ID2LABEL,
    LABEL2ID,
)

from evaluation import compute_metrics


def tokenize_and_align_labels(sentences, labels):
    tokenizer = BertTokenizer.from_pretrained(
        "./model/bert-base-cased",
        trust_remote_code=True,
    )

    tokenized = tokenizer(
        sentences,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    aligned_labels = []
    for i, label in tqdm(enumerate(labels), desc="Tokenizing"):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word = None

        label_ids = []
        for word_id in word_ids:

            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)
            previous_word = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels

    return tokenized


def populate_knowledge_base(train_sentences, train_labels):
    kb = NERKnowledgeBase()
    for words, tags in tqdm(zip(train_sentences, train_labels), desc="Populating knowledge base"):
        entities = extract_entities(words, tags)
        text = " ".join(words)

        for entity, category in entities:
            kb.add_entity(entity, category, texts=[text])


def train():
    print("Starting dataset preprocessing...")
    processor = NERDatasetProcessor(Path("./dataset/train.csv"), split="train")
    processor.load_data()
    processor.clean_data()
    sentences, labels = processor.get_sentences()
    dataset_stats(sentences)
    # Adding training data to knowledge base
    populate_knowledge_base(sentences, labels)

    train_s, val_s, train_l, val_l = train_test_split(
        sentences,
        labels,
        test_size=0.1,
        random_state=42,
    )

    train_tokenized = tokenize_and_align_labels(train_s, train_l)
    val_tokenized = tokenize_and_align_labels(val_s, val_l)
    train_dataset = NERDataset(train_tokenized)
    val_dataset = NERDataset(val_tokenized)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BertForTokenClassification.from_pretrained(
        "./model/bert-base-cased",
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )
    model.to(device)

    training_args = TrainingArguments(
        seed=42,
        data_seed=42,
        output_dir="./model/bert-ner",
        eval_strategy="epoch",
        learning_rate=3e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-8},
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        logging_steps=25,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train()
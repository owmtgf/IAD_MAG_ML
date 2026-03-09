from pathlib import Path
from argparse import ArgumentParser

import torch
from transformers import (
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split

from ..knowledgebase import populate_knowledge_base
from .dataset import (
    NERDatasetProcessor,
    NERDataset,
    dataset_stats,
)
from .model_utils import load_model, load_tokenizer
from .pipeline import tokenize
from .evaluate import compute_metrics
from ..globals import RANDOM_SEED


def train(
        checkpoint_dir: Path,
        input_csv: Path,
        output_dir: Path,
        batch_size: int,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_model(checkpoint_dir)
    model.to(device)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    print("Starting dataset preprocessing...")
    processor = NERDatasetProcessor(input_csv, split="train")
    processor.load_data()
    processor.clean_data()
    sentences, labels = processor.get_sentences()
    dataset_stats(sentences)

    # Adding training data to knowledge base
    kb = populate_knowledge_base(
        knowledgebase=None,
        train_sentences=sentences, 
        train_labels=labels,
    )
    kb.save(output_dir / "knowledge_base" / "kb.json")

    train_s, val_s, train_l, val_l = train_test_split(
        sentences,
        labels,
        test_size=0.1,
        random_state=RANDOM_SEED,
    )

    train_tokenized = tokenize(tokenizer, train_s, train_l)
    val_tokenized = tokenize(tokenizer, val_s, val_l)
    train_dataset = NERDataset(train_tokenized)
    val_dataset = NERDataset(val_tokenized)

    training_args = TrainingArguments(
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=4.8e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1.2e-8},
        warmup_ratio=0.04,
        weight_decay=0.027,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        num_train_epochs=4,
        logging_steps=25,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint_dir", type=Path,
                        default=Path("./checkpoints/bert-large-cased"))
    parser.add_argument("-i", "--input_csv", type=Path,
                        default=Path("./dataset/train.csv"))
    parser.add_argument("-o", "--output_dir", type=Path,
                        default=Path("./checkpoints/bert-ner-test"))
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(f"Start running {__file__}. Running arguments:\n{args}")

    checkpoint_dir = args.checkpoint_dir
    input_csv = args.input_csv
    output_dir = args.output_dir
    batch_size = args.batch_size

    train(
        checkpoint_dir=checkpoint_dir,
        input_csv=input_csv,
        output_dir=output_dir,
        batch_size=batch_size,
    )

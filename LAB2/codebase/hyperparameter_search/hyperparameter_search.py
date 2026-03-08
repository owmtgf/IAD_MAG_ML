import json
from pathlib import Path
from argparse import ArgumentParser

from transformers import (
    TrainingArguments,
    Trainer,
    DistilBertForTokenClassification,
    DataCollatorForTokenClassification,
)
from sklearn.model_selection import train_test_split

from ..model.dataset import (
    NERDatasetProcessor,
    NERDataset,
    dataset_stats,
)
from ..model.pipeline import tokenize
from ..model.evaluate import compute_metrics
from ..globals import ID2LABEL, LABEL2ID


def model_init(model_path: str):
    return DistilBertForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def hp_space(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "lr_scheduler_kwargs": {
            "min_lr": trial.suggest_float("min_lr", 1e-8, 1e-6, log=True)
        },
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),
    }

    return params


def process_dataset(dataset_path: Path):
    processor = NERDatasetProcessor(dataset_path, split="train")
    processor.load_data()
    processor.clean_data()
    sentences, labels = processor.get_sentences()
    dataset_stats(sentences)

    train_s, val_s, train_l, val_l = train_test_split(
        sentences,
        labels,
        test_size=0.1,
        random_state=42,
    )

    train_tokenized, tokenizer = tokenize(train_s, train_l)
    val_tokenized, _ = tokenize(val_s, val_l)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    return NERDataset(train_tokenized), NERDataset(val_tokenized), data_collator


def train(model_path: str, train_dataset_path: Path, output_model_path: str):
    train_dataset, val_dataset, data_collator = process_dataset(train_dataset_path)

    training_args = TrainingArguments(
        output_dir=output_model_path,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        load_best_model_at_end=False,
        lr_scheduler_type="cosine_with_min_lr",
        metric_for_best_model="f1",
        num_train_epochs=1,
    )
    trainer = Trainer(
        model_init=model_init(model_path),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=20,
    )

    return best_trial.hyperparameters


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        default="./checkpoints/distilbert-base-cased",
        help="Path to he distillated model",
    )
    parser.add_argument(
        "--train_dataset_path",
        "-td",
        type=Path,
        default="./dataset/train.csv",
        help="Path to train dataset CSV file",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="./checkpoints/distilbert-hp",
        help="Path to output hyperparameter search model",
    )
    parser.add_argument(
        "--output_hp_file",
        "-ohp",
        type=str,
        default="./hyperparameter_search/best_hyperparameters.json",
        help="Path to output file with best hyperparameters",
    )
    args = parser.parse_args()
    parameters = train(args.model_path, args.train_dataset_path, args.output_model_path)

    with open(args.output_hp_file, "w") as f:
        json.dump(parameters, f, indent=4)
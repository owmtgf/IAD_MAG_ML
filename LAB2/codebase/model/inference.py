from pathlib import Path
from argparse import ArgumentParser

import torch
import pandas as pd
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification

from .dataset import (
    NERDatasetProcessor,
    NERDataset,
)
from .pipeline import tokenize, extract_predictions
from .model_utils import load_tokenizer, load_model


def predict(model, tokenizer, dataset, device, batch_size=32):
    model.eval()
    predictions = []

    data_collator = DataCollatorForTokenClassification(tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())

    return predictions


def save_submission(predictions, output_path):

    df = pd.DataFrame({
        "ID": range(len(predictions)),
        "Tag": predictions
    })

    df.to_csv(output_path, index=False)


def prepare_sentences_csv_from_text(sentences: list[str]) -> None:
    rows = []
    sid = 1

    for sent in sentences:
        words = sent.strip().split()
        for i, word in enumerate(words):
            row = {
                "Sentence_id": float(sid),
                "Word": word,
            }
            rows.append(row)
        sid += 1

    df = pd.DataFrame(rows)
    return df


def run_inference(
    checkpoint_dir: Path,
    input_data: Path | list,
    output_csv: Path = None,
    batch_size: int = 32,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_model(checkpoint_dir)
    model.to(device)

    if isinstance(input_data, list):
        input_data = prepare_sentences_csv_from_text(input_data)

    processor = NERDatasetProcessor(
        input_data,
        split="test"
    )
    processor.load_data()
    processor.clean_data()

    sentences, _ = processor.get_sentences()
    tokenized, word_ids = tokenize(tokenizer, sentences)

    dataset = NERDataset(tokenized)

    predictions = predict(model, tokenizer, dataset, device, batch_size)
    word_predictions = extract_predictions(predictions, word_ids, sentences)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_submission([pred for sent in word_predictions for pred in sent], output_csv)

        print("Expected rows:", sum(len(s) for s in sentences))
        print("Submission rows:", sum(len(s) for s in word_predictions))

    return sentences, word_predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint_dir", type=Path,
                        default=Path("./checkpoints/bert-ner-large/checkpoint-4320"))
    parser.add_argument("-i", "--input_csv", type=Path,
                        default=Path("./dataset/test.csv"))
    parser.add_argument("-o", "--output_csv", type=Path,
                        default=Path("./outputs/submission.csv"))
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(f"Start running {__file__}. Running arguments:\n{args}")

    checkpoint_dir = args.checkpoint_dir
    input_csv = args.input_csv
    output_csv = args.output_csv
    batch_size = args.batch_size

    run_inference(
        checkpoint_dir,
        input_csv,
        output_csv,
        batch_size,
    )

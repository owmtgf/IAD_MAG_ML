# NER Task

## Description
This project implements a Named Entity Recognition (NER) system enhanced with a Knowledge Base (KB) module. It can:
- Train a BERT-based NER model on provided dataset
- Extract and store recognized entities into a structured knowledge base
- Allow manual and automatic addition of entities and categories
- Retrieve contextual information, example texts, and meanings for entities
- Visualize entities with word clouds and highlight entities in sentences
- Support inference on new texts, saving predictions, and populating the KB dynamically

---

## Main Components

### Datasets
- `dataset/train.csv` - training data
- `dataset/test.csv` - test data

### Codebase
- `codebase/model/` - NER model implementation, training, inference, evaluation
- `codebase/hyperparameter_search/` - scripts to search for optimal hyperparameters
- `codebase/knowledgebase.py` - NER Knowledge Base module
- `codebase/pipeline.py` - tokenization, and prediction pipeline

### Analysis
- `data_analysis.ipynb` - data analysis and KB demonstration, found methods are involved in preprocessing in `codebase/model/dataset.py`
- `knowledge_base.ipynb` - Knowledge Base workflow demonstration

### Extras
- `knowledge_base/kb.json` - Knowledge Base saved state collected via train data fitting
- `outputs/submission.csv` - Saved inference results for `dataset/test.csv`

---

## Run Training
```sh
python -m codebase.model.train \
    --checkpoint_dir "./checkpoints/bert-large-cased" \
    --input_csv "./dataset/train.csv" \
    --output_dir "./checkpoints/bert-large-ner" \
    --batch_size 32
```

## Perform Hyperparameter search
```sh
python -m codebase.hyperparameter_search.hyperparameter_search \
    --model_path "./checkpoints/distilbert-base-cased" \
    --train_dataset_path "./dataset/train.csv" \
    --output_model_path "./checkpoints/distilbert-hp" \
    --output_hp_file "./codebase/hyperparameter_search/best_hyperparameters.json"
```
> You can find current best parameters in existing `codebase/hyperparameter_search/best_hyperparameters.json`

## Inference the model
```sh
python -m codebase.model.inference \
    --checkpoint_dir "./checkpoints/bert-ner-large/checkpoint-4320" \
    --input_csv "./dataset/test.csv" \
    --output_csv "./outputs/submission.csv" \
    --batch_size 32
```
> You can find current inference results for `test.csv` in existing `"./outputs/submission.csv"`

Optionally populate Knowledge Base with inference results on new sentences:

```python
from codebase.model.inference import run_inference
from codebase.knowledgebase import NERKnowledgeBase

input_sentences = [
    "Does the name Ibrahim mean anything to you ?"
]

sentences, word_predictions = run_inference(
    checkpoint_dir="./checkpoints/bert-ner-large/checkpoint-4320",
    input_data=input_sentences,
)

kb = NERKnowledgeBase.load("./knowledge_base/kb.json")
# or initialize empty
# kb = NERKnowledgeBase()
kb.add_entities_from_model(sentences, word_predictions)
```

## Evaluate inference results
```sh
python -m codebase.model.evaluate \
    -gt "./dataset/test.csv" \  # suppose `Tag` feature in
    -pred "./outputs/submission.csv" \
    -m "./outputs/metrics.csv"
```
from transformers import BertTokenizerFast, BertForTokenClassification
from ..globals import ID2LABEL, LABEL2ID


def load_tokenizer(model_path):

    return BertTokenizerFast.from_pretrained(
        model_path,
        trust_remote_code=True,
    )


def load_model(model_path):

    return BertForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )

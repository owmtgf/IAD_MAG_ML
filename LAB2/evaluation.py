from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataset import ID2LABEL


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.argmax(axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):

        current_preds = []
        current_labels = []

        for p, l in zip(pred, lab):

            if l != -100:
                current_preds.append(ID2LABEL[p])
                current_labels.append(ID2LABEL[l])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

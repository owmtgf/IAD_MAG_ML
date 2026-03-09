def tokenize(tokenizer, sentences, labels=None, max_length=128):

    tokenized = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length, # 512 accordingly to pre-trained tokenizer config, we force it to 128 due data analysis results 
    )

    if labels is None:
        word_ids = [
            tokenized.word_ids(batch_index=i)
            for i in range(len(sentences))
        ]

        return tokenized, word_ids

    aligned_labels = []

    for i, label in enumerate(labels):
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


def extract_predictions(predictions, word_ids_list, sentences):
    final_preds = []

    for sent_preds, word_ids, words in zip(predictions, word_ids_list, sentences):
        word_preds = [0] * len(words)
        seen = set()

        for pred, word_id in zip(sent_preds, word_ids):
            if word_id is None:
                continue
            if word_id not in seen:
                word_preds[word_id] = int(pred)
                seen.add(word_id)

        final_preds.append(word_preds)

    return final_preds

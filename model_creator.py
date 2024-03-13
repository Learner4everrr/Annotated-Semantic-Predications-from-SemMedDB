from transformers import AutoModelForSequenceClassification, AutoTokenizer


def model_creator(model_name):
    labels = ['n', 'y']
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

    # Freeze all layers except the last one
    for param in model.base_model.parameters():
        param.requires_grad = False

    if model_name == "bert-base-uncased" or model_name == "google-bert/bert-large-uncased":
        for param in model.base_model.pooler.dense.parameters():
            param.requires_grad = True
    elif model_name == "roberta-base":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "distilbert-base-uncased":
        for param in model.distilbert.parameters():
            param.requires_grad = True
    else:
        print("Model is not in the list!")

    return model, tokenizer


if __name__ == '__main__':
    main()

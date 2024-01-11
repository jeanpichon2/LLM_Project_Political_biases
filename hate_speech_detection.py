import pandas as pd
import numpy as np
import json, re
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaConfig,
)
from datasets import Dataset
from tqdm import tqdm_notebook
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

# Load the json file
file = "identity_hate_corpora.jsonl"

dataset = []
with open(file, "r") as f:
    for line in f:
        object_json = json.loads(line)
        dataset.append(object_json)


# Convert the dataset into a Hugging Face Dataset
formatted_dataset = {
    "text": [line["text"] for line in dataset],
    "fold": [line["fold"] for line in dataset],
    "grouping": [line["grouping"] for line in dataset],
    "hate": [line["hate"] for line in dataset],
}

dataset_hf = Dataset.from_dict(formatted_dataset)

# Shuffle the data
data = dataset_hf.shuffle()

n_samples = 100

data = data.select(range(n_samples))

# Split the dataset into train_set and test_set
train_data = data.filter(lambda example: example["fold"] == "train")
test_data = data.filter(lambda example: example["fold"] == "test")

# Remove useless columns
train_data = train_data.select_columns(["text", "hate", "grouping"])
test_data = test_data.select_columns(["text", "hate", "grouping"])

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


def preprocess(text):
    if text == "":
        return ""
    else:
        text = text.lower()
        text_cleaned = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text_cleaned = re.sub(r"#[A-Za-z0-9_]+", "", text_cleaned)
        text_cleaned = re.sub(r"https?:\/\/\S*", "", text_cleaned)
        text_cleaned = text_cleaned.replace(",", "")
    return text_cleaned


# Use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def create_sentence_embeddings(sentences):
    input_ids = []

    for sent in sentences:
        preprocessed_sent = preprocess(sent)
        input = tokenizer.encode_plus(
            preprocessed_sent,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation="longest_first",
        )
        input_ids.append(input["input_ids"])

    input_ids = torch.tensor(input_ids)
    return input_ids


# Parameters
loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(params=model.parameters(), lr=1e-6)


def train(orig_train, model, batch_size=32):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2)

    for train_inds, dev_inds in splitter.split(
        orig_train, groups=range(len(orig_train))
    ):
        train_subset = [orig_train[i] for i in train_inds.tolist()]
        dev_subset = [orig_train[i] for i in dev_inds.tolist()]

    input_ids_train = create_sentence_embeddings(
        [item["text"] for item in train_subset]
    )
    input_ids_dev = create_sentence_embeddings([item["text"] for item in dev_subset])

    labels_train = torch.tensor(
        [item["hate"] for item in train_subset], dtype=torch.long
    )
    labels_dev = torch.tensor([item["hate"] for item in dev_subset], dtype=torch.long)

    input_ids_train = torch.tensor(input_ids_train)
    labels_train = torch.tensor(labels_train)

    input_ids_dev = torch.tensor(input_ids_dev)
    labels_dev = torch.tensor(labels_dev)

    train_dataset = TensorDataset(input_ids_train, labels_train)
    dev_dataset = TensorDataset(input_ids_dev, labels_dev)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

    max_epochs = 3
    model = model.train()
    for epoch in tqdm_notebook(range(max_epochs)):
        print("EPOCH -- {}".format(epoch))
        for i, (sent, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            sent = sent.squeeze(0)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                correct = 0
                total = 0
                all_predicted = []
                all_labels = []
                for sent, label in dev_dataloader:
                    sent = sent.squeeze(0)
                    if torch.cuda.is_available():
                        sent = sent.cuda()
                        label = label.cuda()
                    output = model.forward(sent)[0]
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted.cpu() == label.cpu()).sum()
                    all_predicted.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

                accuracy = 100.00 * correct / total
                f1 = f1_score(all_labels, all_predicted)
                balanced_acc = balanced_accuracy_score(all_labels, all_predicted)

                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}%. F1 Score: {}. BACC: {}%".format(
                        i, loss.item(), accuracy, f1, balanced_acc
                    )
                )
    return model


trained_model = train(train_data, model)


def save_model(model):
    filename = f"roberta_hate_identity_3.pth"
    torch.save(model.state_dict(), filename)


save_model(trained_model)


def test(test_dataset, model, batch_size=32):
    model.eval()

    input_ids_test = create_sentence_embeddings([item["text"] for item in test_dataset])
    labels_test = torch.tensor(
        [item["hate"] for item in test_dataset], dtype=torch.long
    )

    input_ids_test = torch.tensor(input_ids_test)
    labels_test = torch.tensor(labels_test)

    test_dataset = TensorDataset(input_ids_test, labels_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    total = 0
    correct = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for sent, label in test_dataloader:
            sent = sent.squeeze(0)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()

            output = model.forward(sent)[0]
            _, predicted = torch.max(output.data, 1)

            total += label.size(0)
            correct += (predicted.cpu() == label.cpu()).sum()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = 100.00 * correct / total
    f1 = f1_score(all_labels, all_predicted)
    balanced_acc = balanced_accuracy_score(all_labels, all_predicted)

    print(
        "Test Results - Accuracy: {}%. F1 Score: {}. BACC: {}%".format(
            accuracy, f1, balanced_acc
        )
    )
    return all_predicted


all_predicted = test(test_data, model)

# Accuracies per identity groups
test_df = pd.DataFrame(test_data)
test_df["pred"] = all_predicted

grouped_data = test_df.groupby("grouping").apply(
    lambda x: {"accuracy": accuracy_score(x["hate"], x["pred"])}
)

for category, metrics in grouped_data.items():
    print(f"\nCategory: {category}")
    print(f"Accuracy: {metrics['accuracy']:.2f}")

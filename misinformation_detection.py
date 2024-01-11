import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tabulate import tabulate
from tqdm.notebook import tqdm
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
)

df_train = pd.read_csv("/content/train.tsv", sep="\t", header=None)
df_valid = pd.read_csv("/content/valid.tsv", sep="\t", header=None)
df_test = pd.read_csv("/content/test.tsv", sep="\t", header=None)

df_train.head()

from datasets import Dataset
from datasets import DatasetDict


def preprocessing_fn1(df):
    # changing the label to true or false
    df["label"] = [
        1 if x == "true" or x == "mostly-true" or x == "half-true" else 0 for x in df[1]
    ]

    # df.dropna
    df = df.fillna("")

    # we drop the columns of the counts and the id
    df = df.drop([0, 1, 8, 9, 10, 11, 12], axis=1)

    # join the metadata in a single column
    metadata = []

    for i in range(len(df)):
        speaker = df[4][i]
        if speaker == 0:
            speaker = ""

        subject = df[4][i]
        if subject == 0:
            subject = ""

        job = df[5][i]
        if job == 0:
            job = ""

        state = df[6][i]
        if state == 0:
            state = ""

        affiliation = df[7][i]
        if affiliation == 0:
            affiliation = ""

        context = df[13][i]
        if context == 0:
            context = ""

        metadata.append(
            str(subject)
            + " "
            + str(speaker)
            + " "
            + str(job)
            + " "
            + str(state)
            + " "
            + str(affiliation)
            + " "
            + str(context)
        )

    # Adding the metadata column to the dataset
    df[14] = metadata

    # Creating a new column composed of the metadata in front of the sentence
    df["sentence"] = (
        df[14].astype("str") + " " + df[2]
    )  # Combining metadata and the text columns into single columns

    # We drop all columns apart from the label and the sentence
    df = df.drop([2, 3, 4, 5, 6, 7, 13], axis=1)

    # Creating a dictionnary of labels and sentences
    pre_dataset = {
        "label": [df["label"][i] for i in range(df.shape[0])],
        "sentence": [df["sentence"][i] for i in range(df.shape[0])],
    }

    # Transforming the dataframe into a dataset
    dataset = Dataset.from_dict(pre_dataset)

    return dataset


dataset_train = preprocessing_fn1(df_train)
dataset_test = preprocessing_fn1(df_test)
dataset_valid = preprocessing_fn1(df_valid)

print(dataset_train)
print(dataset_test)
print(dataset_valid)

from torch.utils.data import Dataset

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

import re


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


def create_sentence_embeddings(sentences):
    input_ids = []

    for sent in sentences:
        preprocessed_sent = preprocess(sent)
        input = tokenizer.encode_plus(
            preprocessed_sent,
            add_special_tokens=True,
            max_length=200,
            padding="max_length",
            return_attention_mask=False,
            truncation=True,
        )
        input_ids.append(input["input_ids"])

    input_ids = torch.tensor(input_ids)
    return input_ids


## Check if Cuda is Available
print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

from tqdm import tqdm_notebook
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score, balanced_accuracy_score

loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-6
optimizer = torch.optim.RAdam(
    params=model.parameters(), lr=learning_rate, weight_decay=1e-5
)


def train(model, dataset_train, dataset_valid, batch_size=32):
    input_ids_train = create_sentence_embeddings(dataset_train["sentence"])
    input_ids_dev = create_sentence_embeddings(dataset_valid["sentence"])

    labels_train = torch.tensor(dataset_train["label"], dtype=torch.long)
    labels_dev = torch.tensor(dataset_valid["label"], dtype=torch.long)

    input_ids_train = torch.tensor(input_ids_train)
    labels_train = torch.tensor(labels_train)

    input_ids_valid = torch.tensor(input_ids_dev)
    labels_valid = torch.tensor(labels_dev)

    train_dataset = TensorDataset(input_ids_train, labels_train)
    valid_dataset = TensorDataset(input_ids_valid, labels_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    max_epochs = 10
    model = model.train()
    for epoch in tqdm_notebook(range(max_epochs)):
        print("EPOCH -- {}".format(epoch))
        for i, (sent, label) in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                all_predicted = []
                all_labels = []
                correct = 0
                total = 0
                for sent, label in valid_loader:
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

                accuracy = 100.00 * correct.numpy() / total
                f1 = f1_score(all_labels, all_predicted)
                balanced_acc = balanced_accuracy_score(all_labels, all_predicted)

                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}%. F1 Score: {}. BACC: {}".format(
                        i, loss.item(), accuracy, f1, balanced_acc
                    )
                )
    return model


import torch

torch.cuda.empty_cache()

trained_model = train(model, dataset_train, dataset_valid)


def save_model(model):
    filename = f"roberta_misinformation_10epochs.pth"
    torch.save(model.state_dict(), filename)


save_model(trained_model)


def test(trained_model, data, batch_size=32):
    model.eval()

    input_ids = create_sentence_embeddings(data["sentence"])

    labels = torch.tensor(data["label"], dtype=torch.long)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, labels)

    loader = DataLoader(dataset, batch_size=batch_size)

    all_predicted = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for sent, label in loader:
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

    accuracy = 100.00 * correct.numpy() / total
    f1 = f1_score(all_labels, all_predicted)
    balanced_acc = balanced_accuracy_score(all_labels, all_predicted)

    print(
        "Test results - Accuracy: {}%. F1 Score: {}. BACC: {}%".format(
            accuracy, f1, balanced_acc
        )
    )


test(trained_model, dataset_test)

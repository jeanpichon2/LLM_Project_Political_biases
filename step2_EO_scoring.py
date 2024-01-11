from transformers import pipeline, AutoTokenizer
from transformers import pipeline
from tqdm import tqdm
import json
import argparse

# Define the lexicons for agreement and disagreement
agree_lexicons = [
    "agree",
    "agrees",
    "agreed",
    "agreeing",
    "accept",
    "accepts",
    "accepted",
    "accepting",
    "approve",
    "approves",
    "approved",
    "approving",
    "endorse",
    "endorses",
    "endorsed",
    "endorsing",
    "believe",
    "believes",
    "believed",
    "believing",
]
disagree_lexicons = [
    "disagree",
    "disagrees",
    "disagreed",
    "disagreeing",
    "oppose",
    "opposes",
    "opposing",
    "opposed",
    "deny",
    "denies",
    "denying",
    "denied",
    "refuse",
    "refuses",
    "refusing",
    "refused",
    "reject",
    "rejects",
    "rejecting",
    "rejected",
    "disapprove",
    "disapproves",
    "disapproving",
    "disapproved",
]


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m", "--model", help="the language model of interest on HuggingFace"
    )
    argParser.add_argument(
        "-d", "--device", default=-1, help="device ID, -1 for CPU, >=0 for GPU ID"
    )

    args = argParser.parse_args()
    model = args.model
    device = int(args.device)

    statement_file = json.loads(
        open("response/" + model[model.find("/") + 1 :] + ".jsonl", "r").read()
    )
    # print(statement_file[0])

    f = open("score/" + model[model.find("/") + 1 :] + ".txt", "w")

    for i in tqdm(range(len(statement_file))):
        statement = statement_file[i]
        response_list = statement["response"]
        positive = 0
        negative = 0
        for response in response_list:
            token_str = "".join(response["token_str"].split())
            score = response["score"]
            if token_str in agree_lexicons:
                positive += score
            elif token_str in disagree_lexicons:
                negative += score

        f.write(
            str(i) + " agree: " + str(positive) + " disagree: " + str(negative) + "\n"
        )
    f.close()

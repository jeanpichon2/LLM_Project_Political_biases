import json
import argparse
from transformers import pipeline
from tqdm import tqdm

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

    statement_file = json.loads(open("response/example.jsonl", "r").read())
    # print(statement_file[0])

    generator = pipeline(
        "text-generation", model=model, device=device, max_new_tokens=30
    )  # max_new_tokens set at 100 at first

    # MY PROMPTS
    # prompt = "People say that <statement>\nI think"
    # PROMPTS FROM THE ARTICLE
    prompt = "Please respond to the following statement: <statement> \n Your response:"
    # prompt =  "What do you think about the following statement: <statement> \n Your response:"
    # prompt =  "What is your opinion on the following statement: <statement> \n Your response:"
    # prompt =  "How do you feel about the following statement: <statement> \n Your response:"
    # prompt =  "Do you agree or disagree with the following statement: <statement> \n Your response:"
    # prompt =  "What is your take on the following statement: <statement> \n Your response:"
    # prompt =  "Tell us about your thoughts on the following statement: <statement> \n Your response:"

    for i in tqdm(range(len(statement_file))):
        statement = statement_file[i]["statement"]
        result = generator(prompt.replace("<statement>", statement))
        statement_file[i]["response"] = result[0]["generated_text"][
            len(prompt.replace("<statement>", statement)) + 1 :
        ]

    # save in jsonl style with indent 4
    with open("response/" + model[model.find("/") + 1 :] + ".jsonl", "w") as f:
        json.dump(statement_file, f, indent=4)

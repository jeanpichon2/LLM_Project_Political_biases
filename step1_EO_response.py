import json
import argparse
from transformers import pipeline
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="the language model of interest on HuggingFace")
    argParser.add_argument("-d", "--device", default = -1, help="device ID, -1 for CPU, >=0 for GPU ID")

    args = argParser.parse_args()
    model = args.model
    device = int(args.device)

    statement_file = json.loads(open("response/example.jsonl", "r").read())
    # print(statement_file[0])

    generator = pipeline("fill-mask", model = model, device = device)

    # you can adjust the prompt slightly for different language models to better elicit opinions. 
    # ADAPT MASK FOR EACH MODEL [MASK] or <mask>
    #1
    #prompt = "<statement> I <mask> with this statement."
    #2
    prompt = "Please respond to the following statement: <statement> I <mask> with this statement."
    #3
    #prompt = "People say that <statement>\n I <mask> with this statement."
    #4
    #prompt = "<statement>\n I <mask>." 

    for i in tqdm(range(len(statement_file))):
        statement = statement_file[i]["statement"]
        results = generator(prompt.replace("<statement>", statement),top_k = 10)
        #print(results)
        statement_file[i]["response"] = [{'token_str': result['token_str'], 'score': result['score']} for result in results[:10]]
        
    
    # save in jsonl style with indent 4
    with open("response/" + model[model.find('/') + 1:] + ".jsonl", "w") as f:
        json.dump(statement_file, f, indent = 4)
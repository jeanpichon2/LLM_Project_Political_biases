from transformers import pipeline, AutoTokenizer, AutoModel
import argparse
from transformers import AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel


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

    # #For encoder-only such as roberta-base / distilroberta-base  <mask>, bert-base-cased [MASK] (replace mask token)
    # unmasker = pipeline("fill-mask", model = model, device = device)
    # result = unmasker("Some people say that same-sex marriage is natural. I <mask> with this statement.")
    # print(result[0]['token_str'])

    # Works for gpt2
    model = AutoModel.from_pretrained(model)
    generator = pipeline(
        "text-generation", model=model, device=device, max_new_tokens=100
    )
    result = generator("Some people say that same-sex marriage is natural. I think")
    print(result[0]["generated_text"])
    print("success!")

    # # works for "meta-llama/Llama-2-7b-chat-hf" (place token after "huggingface-cli login" with one right click+enter) but too large to load!
    # from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
    # config = LlamaConfig()
    # tokenizer = LlamaTokenizer.from_pretrained(model)
    # model = LlamaForCausalLM.from_pretrained(model, config = config)
    # generator = pipeline("text-generation", model = model, device = device, max_new_tokens = 100)
    # result = generator("Some people say that same-sex marriage is natural. I think")
    # print(result[0]["generated_text"])
    # print("success!")

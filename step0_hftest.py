from transformers import pipeline, AutoTokenizer, AutoModel
import argparse
from transformers import AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="the language model of interest on HuggingFace")
    argParser.add_argument("-d", "--device", default = -1, help="device ID, -1 for CPU, >=0 for GPU ID")

    args = argParser.parse_args()
    model = args.model
    device = int(args.device)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # model = AutoModelForCausalLM.from_pretrained(model)

    # #For encoder-only such as roberta-base / distilroberta-base  <mask>, bert-base-cased [MASK] (replace mask token), also surely bert-large-uncased
    # unmasker = pipeline("fill-mask", model = model, device = device)
    # result = unmasker("Some people say that same-sex marriage is natural. I <mask> with this statement.")
    # print(result[0]['token_str'])

    # Works for gpt2
    model = AutoModel.from_pretrained(model)
    generator = pipeline("text-generation", model = model, device = device, max_new_tokens = 100)
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



    # import TFT5ForConditionalGeneration, TFT5Tokenizer
    # # Load T5 model and tokenizer
    # model = "t5-small"
    # model = TFT5ForConditionalGeneration.from_pretrained(model)
    # tokenizer = TFT5Tokenizer.from_pretrained(model)
    # generator = pipeline("text-generation", model = model, tokenizer=tokenizer, device = device, max_new_tokens = 100)
    # result = generator("Tell me more about Donald Trump")
    # print("generated text :",result[0]["generated_text"])
    # import tensorflow as tf
    # # from transformers import T5Tokenizer

    # # # Load T5 model and tokenizer
    # # model = "t5-small"
    # # tokenizer = T5Tokenizer.from_pretrained(model)
    # # model = tf.keras.models.load_model(model)

    # # # Text to be generated
    # # input_text = "Tell me more about Donald Trump"

    # # # Tokenize input text
    # # input_ids = tokenizer(input_text, return_tensors="tf").input_ids

    # # # Generate text
    # # output = model.generate(input_ids)
    # # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # # print("Generated Text:", generated_text)

    # from transformers import BartTokenizer, BartForConditionalGeneration

    # tokenizer = BartTokenizer.from_pretrained(model)
    # model = BartForConditionalGeneration.from_pretrained(model)

    # input_text = "who is donald trump?"

    # # Tokenize input text
    # input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024)

    # # Generate text
    # output = model.generate(input_ids, max_length=50, num_beams=5, length_penalty=2.0, early_stopping=True)

    # # Decode the generated output
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # print("Generated Text:", generated_text)


    # print("success!")
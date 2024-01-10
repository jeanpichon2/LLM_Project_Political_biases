from transformers import pipeline, AutoTokenizer
from transformers import pipeline
from tqdm import tqdm
import json
import argparse

classifier = pipeline("zero-shot-classification", model = "bert-base-MultiNLI")
response = "I like you. I don't love you."

result = classifier(response, candidate_labels=["agree", "disagree"])
print(result)
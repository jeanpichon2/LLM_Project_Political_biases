#### Step 0: Sanity Check
We mainly implement things with the text generation pipeline of Huggingface Transformers. Check out your HuggingFace model compatibility by running:
```
python step0_hftest.py --model <your_model> --device <your_device>
```
If you see `success!` printed out, you are good to go!

#### Step 1: Generate Responses
If your step 0 is successful, run:
```
python step1_response.py --model <your_model> --device <your_device>
```
Or for encoders-only, run:
```
python step1_EO_response.py --model <your_model> --device <your_device>
```
There should be a jsonl file in `response/` with your model name, containing the generated text for each statement. For encoders-only, it contains the top-10 probable token for mask filling for each statement.
Note that we only prompt once for clarity and efficiency, while the paper used an average of 5 runs.

#### Step 2: Get Agree/Disagree Scores
Run:
```
python step2_scoring.py --model <your_model> --device <your_device>
```
We use here a stance detector to evaluate the political leaning of decoderbased language models. The goal
of stance detection is to judge the LM-generated response and map it to {STRONG DISAGREE, DISAGREE, AGREE, STRONG AGREE}. To this end, we employed the FACEBOOK/BART-LARGE-MNLI checkpoint on Huggingface Transformers.

For encoders-only, run:
```
python step2_EO_scoring.py --model <your_model> --device <your_device>
```
We aggregate the probability of positive and negative words based on lexicons, and set a threshold to map
them to {STRONG DISAGREE, DISAGREE, AGREE, STRONG AGREE}.

There should be a txt file in `score/` with your model name. Each line presents the agree/disagree probabilities for each political statement.

#### Step 3: Get Political Leaning with the Political Compass Test
Run `python step3_testing.py --model <your_model>. The script will automatically open the Chrome browser and take the test. The final political leaning will be displayed on the website. Please note that the browser will first be on the adblocker tab, make sure not to close it and switch to the political compass test tab after the ad blocker is successfully loaded. Adapt the path to your adblocker.

### Pretraining step
Use the `python Pretraining.ipynb` to pretrain any LM on a given dataset. We used it for roberta-base, on smaller sample of reddit-left and reddit-right datasets.


### Misinformation detection
Upload the datasets from the 'liar_dataset' folder and run : 'python misinformation_detection.py'.
The final output will be the performance of the trained RoERTa model on the test dataset.





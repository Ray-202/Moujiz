from flask import Flask, render_template, request

from transformers import BigBirdPegasusForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import pickle

# Prepare the summarization and tagging models
model_name1 = "google/bigbird-pegasus-large-arxiv"
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
model_name2 = "Hatoun/DistiBERT-finetuned-arxiv-multi-label"
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model1 = BigBirdPegasusForConditionalGeneration.from_pretrained(model_name1).to(device)
model2 = AutoModelForSequenceClassification.from_pretrained(model_name2)

# Load the multi-label binarizer
with open(r"/Users/reemyalfaisal/Desktop/Inference02/inference/multi-label-binarizer.pkl", "rb") as f:
    multilabel_binarizer = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]
        # Generate summary
        input_text = "summarize: " + inputtext
        # Set the repetition penalty and length constraint
        repetition_penalty = 2.0
        length_constraint = 4096
        tokenized_text = tokenizer1.encode(input_text,truncation =True, padding ='longest', return_tensors='pt').to(device)
        summary_ = model1.generate(tokenized_text, repetition_penalty=repetition_penalty, max_length=length_constraint)
        summary = tokenizer1.decode(summary_[0])

       # Generate tags
        encoding = tokenizer2(summary, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model2(**encoding)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(outputs.logits[0].cpu())
        preds = np.zeros(probs.shape)
        preds[np.where(probs>=0.3)] = 1 
        # Convert predictions to categories
        tags = multilabel_binarizer.inverse_transform(preds.reshape(1,-1))

        
    


    return render_template("output.html", data = {"summary": summary, "tags": tags})

if __name__ == '__main__': 
    app.run()


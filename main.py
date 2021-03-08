import pandas as pd
import numpy as np
import torch
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

if torch.cuda.is_available():
	device = torch.device("cuda")

else:
	device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("ahotrod/electra_large_discriminator_squad2_512")
model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/electra_large_discriminator_squad2_512", return_dict=False)


def answer_question(question, context):
	inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
	input_ids = inputs["input_ids"].tolist()[0]  # the list of all indices of words in question + context
	text_tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Get the tokens for the question + context
	answer_start_scores, answer_end_scores = model(**inputs)
	# Get the most likely beginning of answer with the argmax of the score
	answer_start = torch.argmax(answer_start_scores)
	# Get the most likely end of answer with the argmax of the score
	answer_end = torch.argmax(answer_end_scores) + 1
	answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
	
	return answer


@app.route('/', methods=['POST', 'GET'])
def index():
	pred = None
	ques = None
	if request.method == 'POST':
		ques = request.form['question']
		cont = request.form['context']
		pred = answer_question(ques, cont)
	
	return render_template('index.html', data=[pred, ques])


if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=False)  # running the app

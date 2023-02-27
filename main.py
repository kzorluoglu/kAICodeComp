
from transformers import pipeline
from flask import Flask
from transformers import AutoTokenizer, RobertaForCausalLM

import json

app = Flask(__name__)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/<phpcode>')
def print_suggestion(phpcode=None):
    fill_mask = pipeline(
        "fill-mask",
        model="huggingface/CodeBERTa-small-v1",
        tokenizer="huggingface/CodeBERTa-small-v1"
    )

    json_results = fill_mask(phpcode)

    json_string = json.dumps(json_results)
    result = json.loads(json_string)

    for item in result:
        del item['token']

    json_result_modified = json.dumps(result)
    response = app.response_class(
        response=json_result_modified,
        status=200,
        mimetype='application/json'
    )
    return response

#
# @app.route('/v2/<phpcode>')
# def print_suggestion_v2(phpcode=None):
#     tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
#     model = RobertaForCausalLM.from_pretrained("huggingface/CodeBERTa-small-v1", is_decoder=True)
#
#     inputs = tokenizer.encode_plus(phpcode, return_tensors="pt", padding=True, max_length=256, truncation=True)
#
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#
#     outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, num_beams=5,
#                              early_stopping=True)
#
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
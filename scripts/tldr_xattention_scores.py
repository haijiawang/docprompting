from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
import torch

device = 'cuda:0'
base_model_name = 'neulab/docprompting-tldr-gpt-neo-125M'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

with open('data/tldr/fid.cmd_dev.codet5.t10.json', 'r') as f:
    examples = json.load(f)

print(len(examples))
print(model)

NUM_EXAMPLES = 50 
for i in range(NUM_EXAMPLES): 
    """
    take some input samples, get top k retrievals 
    for each retrieval: 
        1) pass it through the model 
        2) get a) cross attention score, b) pass accuracy 
        3) check correlation between 2 a) and b)
    """
    example = examples[i]
    print(example)
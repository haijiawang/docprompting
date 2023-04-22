from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
import torch

device = 'cuda:0'
base_model_name = 'neulab/docprompting-tldr-gpt-neo-125M'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = model.to(device)

with open('data/conala/fid.cmd_dev.codet5.t10.json', 'r') as f:
    examples = json.load(f)

with open('data/conala/retrieval_results.json', 'r') as f: 
    retrievals = json.load(f) 

NUM_EXAMPLES = 50
K = 10 # look at top K retrievals  
results = {}

for i in range(NUM_EXAMPLES): 
    example = examples[i]
    id = example['id']
    retrieval = retrievals[id] # a dictionary with 'retrieved' and 'scores'

    results[id] = {} 
    print(i, '/', NUM_EXAMPLES)

    for j in range(0, K): 
        print(j, '/', K)
        retrieved_txt = retrieval['retrieved'][j]
        retrieved_score = retrieval['score'][j]

        manual_list = [doc['text'] for doc in example['ctxs']]
        manual_list.append(retrieved_txt) # APPENDING retrieved text here 
        manual_list = "\n".join(manual_list).strip()
        nl = example['question']
        prompt = f'{tokenizer.bos_token} {manual_list} '
        prompt += f'{tokenizer.sep_token} {nl} {tokenizer.sep_token}'

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        if input_ids.shape[1] > 2000: continue
        gen_tokens = model.generate(
            input_ids,
            num_beams=5,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True
        )
        # results[id]['attentions'] = gen_tokens.attentions 

        gen_tokens = gen_tokens.sequences
        gen_tokens = gen_tokens.reshape(1, -1, gen_tokens.shape[-1])[0][0]
        gen_text = tokenizer.decode(gen_tokens)
        # parse
        gen_text = gen_text.split(tokenizer.sep_token)[2].strip().split(tokenizer.eos_token)[0].strip()

        print('===== GEN TEXT =====')
        print(gen_text)
        print('\n')

# with open('scripts/results.json', "w+") as f: 
#     json.dump(results, f, indent=2)


    
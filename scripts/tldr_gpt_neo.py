from transformers import AutoModelForCausalLM, AutoTokenizer
import json

device = 'cuda:0'
base_model_name = 'neulab/docprompting-tldr-gpt-neo-125M'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = model.to(device)

with open('data/tldr/fid.cmd_dev.codet5.t10.json', 'r') as f:
    examples = json.load(f)

idx = 0
for example in examples:
    print(idx, '/', len(examples))
    idx += 1
    manual_list = [doc['text'] for doc in example['ctxs']]
    manual_list = "\n".join(manual_list).strip()
    nl = example['question']
    prompt = f'{tokenizer.bos_token} {manual_list} '
    prompt += f'{tokenizer.sep_token} {nl} {tokenizer.sep_token}'
    print(prompt)
    print('\n=================')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    if input_ids.shape[1] > 2000: continue
    gen_tokens = model.generate(
        input_ids,
        num_beams=5,
        max_new_tokens=150,
        num_return_sequences=2,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_tokens = gen_tokens.reshape(1, -1, gen_tokens.shape[-1])[0][0]
    gen_text = tokenizer.decode(gen_tokens)
    # parse
    gen_text = gen_text.split(tokenizer.sep_token)[2].strip().split(tokenizer.eos_token)[0].strip()
    print(gen_text)

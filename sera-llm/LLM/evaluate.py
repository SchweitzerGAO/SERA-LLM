from collections import Counter
import re
import string
import random
from typing import List

from datasets import load_dataset
from chatgpt import RagChatGPT, model
from base_llm import RagBaseLLM
import time

dataset = load_dataset("hotpot_qa",'distractor',split='validation', cache_dir='../../hotpot-qa')

def get_prompts(indices: int):
    return [dataset[i]['question'] for i in indices]

def get_gts(indices: List[int]):
    return [dataset[i]['answer'] for i in indices]

def get_doc_list(i: int):
    return [sent.strip() for sent_list in dataset[i]['context']['sentences'] for sent in sent_list]

def generate_idx(n: int):
    return random.sample(range(0, len(dataset)), n)

def postprocess(output: str):
    # To lower
    output = output.lower()

    # Exclude the punctuations
    exclude = set(string.punctuation)
    output = ''.join(ch for ch in output if ch not in exclude)

    # Remove articles
    output = re.sub(r'\b(a|an|the)\b', ' ', output)

    # Fix spaces
    return ' '.join(output.split())


def f1_score(output, gt):
    common = Counter(output) & Counter(gt)

    # Same words
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    # precision
    precision = 1.0 * num_same / len(output)

    # recall
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(llm: RagBaseLLM, 
             n: int = 100,
             search: bool = False,
             raw: bool = False):
    indices = generate_idx(n)
    prompts = get_prompts(indices)
    gts = get_gts(indices)
    f1 = 0.0
    for idx, prompt, gt in zip(indices, prompts, gts):
        docs_list = None
        if not search:
            docs_list = get_doc_list(idx)
        output = llm.evaluation_hotpot(prompt=prompt, docs_list=docs_list)
        processed_output = postprocess(output)
        processed_gt = postprocess(gt)
        f1 += f1_score(processed_output, processed_gt)
        print(f'Question #{idx} finished')
        time.sleep(3)
    print(f1 / len(gt))
        
        

llm = RagChatGPT(model=model, lang='en')

evaluate(llm)



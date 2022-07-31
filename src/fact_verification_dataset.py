import json

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import T5Tokenizer

model_name = "allenai/unifiedqa-v2-t5-large-1251000"
tokenizer = T5Tokenizer.from_pretrained(model_name)


class MarriageFactVerificationDataset(Dataset):
    def __init__(self, filepath: str):
        self.facts = json.load(open(filepath, 'r'))

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index) -> T_co:
        fact = self.facts[index]
        person_one, person_two = fact[0]
        output = 'yes' if fact[2] else 'no'
        evidence = fact[3].replace('\n', ' ')
        fact_evidence = f'Is {person_one} married to {person_two}?\n{evidence}'
        return {"input": fact_evidence, "output": output}

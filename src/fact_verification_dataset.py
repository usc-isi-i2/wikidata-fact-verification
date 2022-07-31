import json

from torch.utils.data import Dataset


class MarriageFactVerificationDataset(Dataset):
    def __init__(self, filepath: str):
        self.facts = json.load(open(filepath, 'r'))

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        fact = self.facts[index]
        person_one, person_two = fact[0]
        output = 'yes' if fact[2] else 'no'
        evidence = fact[3].replace('\n', ' ')
        fact_evidence = f'Is {person_one} married to {person_two}?\n{evidence}'
        return {"input": fact_evidence, "output": output}

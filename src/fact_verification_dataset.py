import json

from torch.utils.data import Dataset

from src.anchor_templates import generate_spouse_data


class MarriageFactVerificationDataset(Dataset):
    def __init__(self, filepath: str = None):
        self.facts = json.load(open(filepath, 'r')) if filepath else generate_spouse_data()

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        return self.facts[index]

    def summary(self):
        pos_fact_count = len([1 for fact in self.facts if fact['output'] == 'yes'])
        neg_fact_count = len([1 for fact in self.facts if fact['output'] == 'no'])

        return pos_fact_count, neg_fact_count

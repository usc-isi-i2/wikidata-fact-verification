import json

from torch.utils.data import Dataset


class MarriageFactVerificationDataset(Dataset):
    def __init__(self, filepath: str):
        self.facts = json.load(open(filepath, 'r'))

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, index):
        return self.facts[index]

    def summary(self):
        pos_fact_count = len([1 for fact in self.facts if fact['output'] == 'yes'])
        neg_fact_count = len([1 for fact in self.facts if fact['output'] == 'no'])

        return pos_fact_count, neg_fact_count

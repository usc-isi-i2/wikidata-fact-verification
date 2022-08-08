import json
import os.path
from random import shuffle

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TacredSpouseFactVerificationEmptyDataset(Dataset):
    def __init__(self, data_directory, data_file):
        self.rel_ids = json.load(open(os.path.join(data_directory, 'rel2id.json')))
        self.rel_description = {}
        with open(os.path.join(data_directory, 'temp.txt')) as f:
            for line in f:
                description = line.strip().split('\t')
                self.rel_description[description[1]] = description[2:]

        self.data = []
        with open(os.path.join(data_directory, data_file)) as f:
            for line in f:
                data = json.loads(line)
                # Handle only personal relations
                if 'per' not in data['relation']:
                    continue

                head_text = data['h']['name']
                tail_text = data['t']['name']
                # desc_text = ' '.join([t for t in self.rel_description[data["relation"]] if t != "'s"][1:-1])
                question_1 = f'Is {head_text} married with {tail_text}?'
                question_2 = f'Is {tail_text} married with {tail_text}?'
                if data['relation'] == 'per:spouse':
                    self.data.append({'input': f'{question_1}\n', 'output': 'no'})
                    self.data.append({'input': f'{question_2}', 'output': 'no'})

        shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def summary(self):
        pos_fact_count = len([1 for fact in self.data if fact['output'] == 'yes'])
        neg_fact_count = len([1 for fact in self.data if fact['output'] == 'no'])

        return pos_fact_count, neg_fact_count


if __name__ == '__main__':
    dataset = TacredSpouseFactVerificationDataset('../data/tacred', 'train.txt')
    print('Train dataset: ', dataset.summary())

    dataset = TacredSpouseFactVerificationDataset('../data/tacred', 'val.txt')
    print('Eval dataset: ', dataset.summary())

    dataset = TacredSpouseFactVerificationDataset('../data/tacred', 'test.txt')
    print('Test dataset: ', dataset.summary())

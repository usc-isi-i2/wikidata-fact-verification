import json
import os.path

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TacredDataset(Dataset):
    def __init__(self, data_directory, data_file):
        self.rel_ids = json.load(open(os.path.join(data_directory, 'rel2id.json')))
        self.data = []
        self.rel_description = {}
        with open(os.path.join(data_directory, 'temp.txt')) as f:
            for line in f:
                description = line.strip().split('\t')
                self.rel_description[description[1]] = description[2:]

        with open(os.path.join(data_directory, data_file)) as f:
            for line in f:
                data = json.loads(line)
                # Handle only personal relations
                if 'per' not in data['relation']:
                    continue

                evidence = ' '.join(data["token"])
                head_text = data['h']['name']
                tail_text = data['t']['name']
                desc_text = ' '.join(self.rel_description[data["relation"]][1:-1])
                question = f'What is relation between {head_text} and {tail_text}?'
                self.data.append({'input': f'{question}\n{evidence}', 'output': f'{head_text} {desc_text} {tail_text}.'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def summary(self):
        return 'multi_label_dataset', 'multi_label_dataset'


if __name__ == '__main__':
    dataset = TacredDataset('../data/tacred', 'train.txt')
    print(dataset.__getitem__(0))

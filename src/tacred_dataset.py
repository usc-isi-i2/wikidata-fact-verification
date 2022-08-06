import json
import os.path

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TacredDataset(Dataset):
    def __init__(self, data_directory, data_file):
        self.rel_ids = json.load(open(os.path.join(data_directory, 'rel2id.json')))
        self.rel_description = {}
        with open(os.path.join(data_directory, 'temp.txt')) as f:
            for line in f:
                description = line.strip().split('\t')
                self.rel_description[description[1]] = description[2:]

        self.data = open(os.path.join(data_directory, data_file)).readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        print(self.data[index])
        data = json.loads(self.data[index])
        evidence = ' '.join(data["token"])
        head_text = data['h']['name']
        tail_text = data['t']['name']
        desc_text = ' '.join(self.rel_description[data["relation"]][1:-1])
        question = f'What is relation between {head_text} and {tail_text}?'
        return {'input': f'{question}\n{evidence}', 'output': f'{head_text} {desc_text} {tail_text}.'}


if __name__ == '__main__':
    dataset = TacredDataset('../data/tacred', 'train.txt')
    print(dataset.__getitem__(0))

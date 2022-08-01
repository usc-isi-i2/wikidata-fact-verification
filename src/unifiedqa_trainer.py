import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import LOG_FILE

max_source_length = 512
max_target_length = 128


class UnifiedQATrainer:
    def __init__(self, model, tokenizer, train_dataset, evaluation_dataset, optimizer, lr_schedular, device, train_batch_size, eval_batch_size):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.evaluation_dataset = evaluation_dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular
        self.device = device

    def train(self, epoch):
        total_loss = 0
        print(f'Starting training: epoch {epoch}')
        dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size)
        for batch in tqdm(dataloader):
            encoding = self.tokenizer(batch['input'], padding="longest", max_length=max_source_length, truncation=True, return_tensors="pt")
            input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

            target_encoding = self.tokenizer(batch['output'], padding="longest", max_length=max_target_length, truncation=True)
            labels = target_encoding.input_ids
            # replace padding token id's of the labels by -100, so it's ignored by the loss
            labels = torch.tensor(labels)
            labels[labels == self.tokenizer.pad_token_id] = -100

            loss = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device)).loss
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()
            self.lr_schedular.step()
            self.optimizer.zero_grad()

        print(f'\nTotal loss: epoch {epoch}: {total_loss}')

    def evaluate(self, epoch, dataset_name, dataset):
        print(f'Evaluating dataset: {dataset_name}')
        total, correct, tp, tn, fp, fn = len(dataset), 0, 0, 0, 0, 0

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.eval_batch_size)
        for batch in tqdm(dataloader):
            encoding = self.tokenizer(batch['input'], padding="longest", max_length=max_source_length, truncation=True, return_tensors="pt")
            input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

            res = self.model.generate(input_ids.to(self.device))
            predictions = self.tokenizer.batch_decode(res, skip_special_tokens=True)

            correct += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred])
            tp += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred == 'yes'])
            tn += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred == 'no'])
            fp += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == 'no' and pred == 'yes'])
            fn += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == 'yes' and pred == 'no'])

        precision = tp / (tp + fp) if (tp + fp) else 'Error: tp + fp is 0'
        recall = tp / (tp + fn) if (tp + fn) else 'Error: tp + fn is 0'
        f1 = (2 * precision * recall) / (precision + recall) if (precision and recall) else 'Error: Precision or/and Recall is 0'
        accuracy = correct / total
        print(f'\nprecision: {precision}, recall: {recall}, F1: {f1}, accuracy: {accuracy}')
        with open(LOG_FILE, 'a') as f:
            f.write(f'{dataset_name}\t{epoch}\t{precision}\t{recall}\t{f1}\t{accuracy}\n')

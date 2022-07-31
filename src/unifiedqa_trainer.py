import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

max_source_length = 512
max_target_length = 128


class UnifiedQATrainer:
    def __init__(self, model, tokenizer, train_dataset, evaluation_dataset, optimizer, lr_schedular, device='cpu', batch_size=1):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.evaluation_dataset = evaluation_dataset
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.evaluation_dataloader = DataLoader(evaluation_dataset, shuffle=False, batch_size=batch_size)
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular
        self.device = device

    def train(self, epoch):
        total_loss = 0
        print(f'Starting training: epoch {epoch}')
        for batch in tqdm(self.train_dataloader):
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

    def evaluate(self, dataset_name, dataloader):
        print(f'Evaluating dataset: {dataset_name}')
        total, correct, tp, tn, fp, fn = len(dataloader.dataset), 0, 0, 0, 0, 0
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

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        accuracy = correct / total
        print(f'\nprecision: {precision}, recall: {recall}, F1: {f1}, accuracy: {accuracy}')

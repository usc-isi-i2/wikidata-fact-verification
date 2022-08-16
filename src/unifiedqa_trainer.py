from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.utils import delete_dir

max_source_length = 512
max_target_length = 128


class UnifiedQATrainer:
    def __init__(self, run_files, model, tokenizer, optimizer, lr_schedular, device, eval_batch_size):
        self.run_files = run_files
        self.model = model
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular
        self.device = device
        self.best_score = defaultdict(lambda: 0)

    def train(self, dataloader, epoch):
        total_loss = 0
        print(f'Starting training: epoch {epoch}')
        for batch in tqdm(dataloader):
            encoding = self.tokenizer(batch['input'], padding="longest", max_length=max_source_length, truncation=True,
                                      return_tensors="pt")
            input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

            target_encoding = self.tokenizer(batch['output'], padding="longest", max_length=max_target_length,
                                             truncation=True)
            labels = target_encoding.input_ids
            # replace padding token id's of the labels by -100, so it's ignored by the loss
            labels = torch.tensor(labels)
            labels[labels == self.tokenizer.pad_token_id] = -100

            loss = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                              labels=labels.to(self.device)).loss
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()
            self.lr_schedular.step()
            self.optimizer.zero_grad()

        print(f'\nTotal loss: epoch {epoch}: {total_loss}')

    def evaluate(self, epoch, dataset_name, dataset, logfile):
        print(f'Evaluating dataset: {dataset_name}')
        total, correct, tp, tn, fp, fn = len(dataset), 0, 0, 0, 0, 0

        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.eval_batch_size)
        predictions_to_save = []
        for batch in tqdm(dataloader):
            encoding = self.tokenizer(batch['input'], padding="longest", max_length=max_source_length, truncation=True,
                                      return_tensors="pt")
            input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

            res = self.model.generate(input_ids.to(self.device))
            predictions = self.tokenizer.batch_decode(res, skip_special_tokens=True)

            if 'marriages' in dataset_name:
                correct, fn, fp, tn, tp = self.evaluate_helper_marriages(batch, correct, fn, fp, predictions, tn, tp)
            else:
                correct, fn, fp, tn, tp = self.evaluate_helper(batch, correct, fn, fp, predictions, tn, tp)

            if dataset_name != 'train':
                for ip, label, pred in zip(batch['input'], batch['output'], predictions):
                    predictions_to_save.append((ip, label, pred, label == pred))

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision and recall) else 0
        accuracy = correct / total
        print(f'\ntp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')
        score_string = f'\nprecision: {precision}, recall: {recall}, F1: {f1}, accuracy: {accuracy}'
        print(score_string)
        with open(logfile, 'a') as f:
            f.write(f'{dataset_name}\t{epoch}\t{precision}\t{recall}\t{f1}\t{accuracy}\n')

        results_df = pd.DataFrame(predictions_to_save,
                                  columns=['Input', 'Correct', 'Prediction', 'Is Correct prediction'])
        results_df.to_csv(
            f'{self.run_files}/predictions_{dataset_name}_{"pretrained" if epoch == -1 else epoch}.csv', index=False)
        if epoch == -1 or 'train' in dataset_name:
            self.best_score[dataset_name] = f1
            return

        if f1 > self.best_score[dataset_name]:
            save_path = f'{self.run_files}/fine_tuned_model_{dataset_name}'
            self.best_score[dataset_name] = f1
            delete_dir(save_path)
            print(f'Saving best model on {dataset_name} at epoch {epoch} with F1: {f1}')
            self.model.save_pretrained(save_path)
            with open(f'{save_path}/score.txt', 'w') as f:
                f.write(f'Epoch: {epoch}\n')
                f.write(score_string)

    def evaluate_helper(self, batch, correct, fn, fp, predictions, tn, tp):
        correct += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred])
        tp += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred == 'yes'])
        tn += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred == 'no'])
        fp += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == 'no' and pred == 'yes'])
        fn += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == 'yes' and pred == 'no'])
        return correct, fn, fp, tn, tp

    def evaluate_helper_marriages(self, batch, correct, fn, fp, predictions, tn, tp):
        correct += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred])
        tp += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred != '<no answer>'])
        tn += len([1 for actual, pred, in zip(batch['output'], predictions) if actual == pred == '<no answer>'])
        fp += len([1 for actual, pred, in zip(batch['output'], predictions)
                   if actual == '<no answer>' and pred != '<no answer>'])
        fn += len(
            [1 for actual, pred, in zip(batch['output'], predictions) if
             actual != '<no answer>' and pred == '<no answer>'])
        return correct, fn, fp, tn, tp

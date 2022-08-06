import argparse

from accelerate import Accelerator
from transformers import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdafactorSchedule

from src.fact_verification_dataset import MarriageFactVerificationDataset
from src.tacred_dataset import TacredDataset
from src.unifiedqa_trainer import UnifiedQATrainer
from src.utils import prepare_run_files_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train T5 unifiedQA model')
    parser.add_argument('--model_size', type=str, help='T5 model size: small/base/large/3b/11b')
    parser.add_argument('--train_dataset', type=str, help='Train dataset type: train filename in data/unifiedQA (excluding .json)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Eval batch size')
    args = parser.parse_args()

    run_files = './run_files'
    prepare_run_files_directory(run_files)
    logfile = f'{run_files}/model-{args.model_size}_train-{args.train_dataset}_train-batch-{args.train_batch_size}_epochs-{args.epochs}.log'

    accelerator = Accelerator()
    device = accelerator.device
    print(f'Running on device: {device}')

    model_name = f'allenai/unifiedqa-v2-t5-{args.model_size}-1251000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    train_file = f'./data/unifiedQA/{args.train_dataset}.json'
    evaluation_file = './data/unifiedQA/test.json'
    datasets = {
        'train': MarriageFactVerificationDataset(train_file),
        'tacred_train': TacredDataset('./data/tacred', 'train.txt'),
        'tacred_eval': TacredDataset('./data/tacred', 'val.txt'),
        'tacred_test': TacredDataset('./data/tacred', 'test.txt'),
        'eval': MarriageFactVerificationDataset(evaluation_file),
        'common_sense': MarriageFactVerificationDataset()
    }

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    trainer = UnifiedQATrainer(run_files, model, tokenizer, optimizer, lr_scheduler, device, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)
    config_string = f'Run configurations: model={model_name} train={train_file} eval={evaluation_file} train_batch={args.train_batch_size} eval_batch={args.eval_batch_size} epochs={args.epochs}'

    with open(logfile, 'w') as f:
        f.write(config_string + '\n')
        f.write(f'Dataset\tEpoch\tPrecision\tRecall\tF1\tAccuracy\n')

    print('-' * 50)
    print(config_string)
    print('*' * 50)
    for dataset_name, dataset in datasets.items():
        pos_count, neg_count = dataset.summary()
        print(f'Dataset: {dataset_name} pos_count: {pos_count} neg_count: {neg_count}')
    print('*' * 50)
    print(f'Pre fine-tuning evaluations:')
    for dataset_name, dataset in datasets.items():
        if dataset_name in ['eval', 'common_sense']:
            trainer.evaluate(-1, dataset_name, dataset, logfile)
            print('-' * 10)

    for epoch in range(args.epochs):
        print('-' * 50)
        trainer.train(datasets['train'], epoch)
        trainer.train(datasets['tacred_train'], epoch)
        print('-' * 10)
        for dataset_name, dataset in datasets.items():
            trainer.evaluate(epoch, dataset_name, dataset, logfile)
            print('-' * 10)

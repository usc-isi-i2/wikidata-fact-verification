import argparse

from accelerate import Accelerator
from transformers import Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdafactorSchedule

from src.fact_verification_dataset import MarriageFactVerificationDataset
from src.unifiedqa_trainer import UnifiedQATrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train T5 unifiedQA model')
    parser.add_argument('--model_size', type=str, help='T5 model size: small/base/large/3b/11b')
    parser.add_argument('--train_dataset', type=str, help='Train dataset type: train filename in data/unifiedQA (excluding .json)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Train dataset type: train_small/train/train_direct/train_extra_neg/train_extra_pos')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    print(f'Running on device: {device}')

    model_name = f'allenai/unifiedqa-v2-t5-{args.model_size}-1251000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    train_file = f'./data/unifiedQA/{args.train_dataset}.json'
    train_dataset = MarriageFactVerificationDataset(train_file)

    evaluation_file = './data/unifiedQA/test.json'
    evaluation_dataset = MarriageFactVerificationDataset(evaluation_file)

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    trainer = UnifiedQATrainer(model, tokenizer, train_dataset, evaluation_dataset, optimizer, lr_scheduler, device, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)

    config_string = f'Run configurations: model={model_name} train={train_file} eval={evaluation_file} train_batch={args.train_batch_size} eval_batch={args.eval_batch_size} epochs={args.epochs}'
    logfile = f'model-{args.model_size}_train-{args.train_dataset}_train-batch-{args.train_batch_size}_epochs-{args.epochs}.log'
    with open(logfile, 'w') as f:
        f.write(config_string + '\n')
        f.write(f'Dataset\tEpoch\tPrecision\tRecall\tF1\tAccuracy\n')

    print('-' * 50)
    print(config_string)
    print('-' * 50)
    print(f'Pre fine-tuning evaluations:')
    trainer.evaluate(-1, 'train', trainer.train_dataset, logfile)
    print('-' * 10)
    trainer.evaluate(-1, 'eval', trainer.evaluation_dataset, logfile)
    print('-' * 50)

    for epoch in range(args.epochs):
        print('-' * 50)
        trainer.train(epoch)
        print('-' * 10)
        trainer.evaluate(epoch, 'train', trainer.train_dataset, logfile)
        print('-' * 10)
        trainer.evaluate(epoch, 'eval', trainer.evaluation_dataset, logfile)

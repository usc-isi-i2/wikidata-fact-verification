from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Adafactor
from transformers.optimization import AdafactorSchedule

from src.fact_verification_dataset import MarriageFactVerificationDataset
from src.unifiedqa_trainer import UnifiedQATrainer

accelerator = Accelerator()
device = accelerator.device
print(f'Running on device: {device}')

model_name = "allenai/unifiedqa-v2-t5-large-1251000"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

train_dataset = MarriageFactVerificationDataset('./data/unifiedQA/train.json')
evaluation_dataset = MarriageFactVerificationDataset('./data/unifiedQA/test.json')

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

trainer = UnifiedQATrainer(model, tokenizer, train_dataset, evaluation_dataset, optimizer, lr_scheduler, device, 6)

print('-' * 50)
print(f'Pre fine-tuning evaluations:')
trainer.evaluate('train', trainer.train_dataloader)
print('-' * 10)
trainer.evaluate('eval', trainer.evaluation_dataloader)
print('-' * 50)

for epoch in range(5):
    print('-' * 50)
    trainer.train(epoch)
    print('-' * 10)
    trainer.evaluate('train', trainer.train_dataloader)
    print('-' * 10)
    trainer.evaluate('eval', trainer.evaluation_dataloader)

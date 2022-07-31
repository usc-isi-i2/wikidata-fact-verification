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

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
my_dataset = MarriageFactVerificationDataset('./data/ground_truth/spouse_fact_validation_gt.json')
train_dataloader = DataLoader(my_dataset, shuffle=True, batch_size=1)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

trainer = UnifiedQATrainer(model, tokenizer, my_dataset, my_dataset, optimizer, lr_scheduler)

for epoch in range(5):
    print('-' * 50)
    trainer.train(epoch)
    trainer.evaluate('train', trainer.train_dataloader)

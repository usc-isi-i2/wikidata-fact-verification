import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Adafactor
from transformers.optimization import AdafactorSchedule
from src.fact_verification_dataset import MarriageFactVerificationDataset
import json

accelerator = Accelerator()
device = accelerator.device
print(f'Running on device: {device}')

model_name = "allenai/unifiedqa-v2-t5-large-1251000"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
my_dataset = MarriageFactVerificationDataset('/content/spouse_fact_validation_gt.json')
train_dataloader = DataLoader(my_dataset, shuffle=True, batch_size=1)

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

max_source_length = 512
max_target_length = 128


def validate(evidence, claim, **generator_args):
    input_ids = tokenizer.encode(claim + '\n' + evidence, return_tensors="pt")
    input_ids = input_ids.to(device)
    res = model.generate(input_ids, **generator_args)
    answer = tokenizer.batch_decode(res, skip_special_tokens=True)
    for word in answer:
        if word != "yes" and word != "no":
            print(word)

    assert len(answer) == 1

    return True if answer[0] == 'yes' else False


def get_prediction(filename):
    print(f'Processing file: {filename}')
    with open(filename) as f:
        data = json.load(f)

    results = []
    results_both_same = 0

    tp, tn, fp, fn = 0, 0, 0, 0
    for spouse_data in tqdm(data):
        person_one, person_two = spouse_data[0]
        claim_1 = f' is {person_one} married to {person_two}?'
        claim_2 = f'is {person_two} married to {person_one}?'

        correct = spouse_data[2]
        evidence = spouse_data[3]
        result_1 = validate(evidence, claim_1)
        results.append((claim_1, evidence, correct, result_1))
        tp += 1 if (result_1 == correct == True) else 0
        tn += 1 if (result_1 == correct == False) else 0
        fp += 1 if (result_1 == True and correct == False) else 0
        fn += 1 if (result_1 == False and correct == True) else 0

        result_2 = validate(evidence, claim_2)
        results.append((claim_2, evidence, correct, result_2))
        tp += 1 if (result_2 == correct == True) else 0
        tn += 1 if (result_2 == correct == False) else 0
        fp += 1 if (result_2 == True and correct == False) else 0
        fn += 1 if (result_2 == False and correct == True) else 0

        results_both_same += 1 if (result_1 == result_2) else 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('Done')
    print(f'tp={tp},fp={fp},tn={tn},fn={fn}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {(2 * precision * recall) / (precision + recall)}')
    print(f'Count of both same result: {results_both_same}')

    return results

get_prediction('/content/spouse_fact_validation_gt.json')
for epoc in range(10):
    total_loss = 0
    correct = 0
    total = len(my_dataset)
    for batch in tqdm(train_dataloader):
        encoding = tokenizer(
            batch['input'],
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = tokenizer(
            batch['output'], padding="longest", max_length=max_target_length, truncation=True
        )
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = torch.tensor(labels)
        labels[labels == tokenizer.pad_token_id] = -100

        # forward pass
        loss = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                     labels=labels.to(device)).loss

        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f'Total loss: {total_loss}')
    get_prediction('/content/spouse_fact_validation_gt.json')

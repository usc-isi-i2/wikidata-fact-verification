# wikidata-fact-verification

## Data extraction and preparation
Prefer to the notebooks. 

## To finetune the model:
```
python train.py --model_size=small --train_dataset=train_direct_500 --train_batch_size=1 --eval_batch_size=1
```

Saves new best model at an epoch in `/data` and corresponding evaluation prediction in `/data/predicted`.

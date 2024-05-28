import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from opacus import PrivacyEngine

np.random.seed(100)
torch.manual_seed(100)
device = 'cuda'

import warnings
warnings.simplefilter("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rotten_tomatoes'], default='sst2')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--noise_multiplier', type=float, default=0.01)
    parser.add_argument('--clipping_bound', type=float, default=1.0)
    parser.add_argument('--bert_path', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    seq_key = 'text' if args.dataset == 'rotten_tomatoes' else 'sentence'
    num_labels = 2
    
    model = AutoModelForSequenceClassification.from_pretrained(args.bert_path, num_labels=num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, use_fast=True)
    
    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola' or args.dataset == 'sst2':
        metric = load_metric('matthews_correlation')
        train_metric = load_metric('matthews_correlation')
    else:
        metric = load_metric('accuracy')
        train_metric = load_metric('accuracy')

    f_metric = load_metric('f1')  # New added metric
    f_train_metric = load_metric('f1')  # New added metric

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset)
    else:
        datasets = load_dataset(args.dataset)
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    else:
        assert False
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    opt = AdamW(model.parameters(), lr=5e-5)
    
    # Freeze positional embedding layer.
    for name, param in model.named_parameters():
        if 'position_embeddings' in name:
            param.requires_grad = False
    
    model.train()
    privacy_engine = PrivacyEngine()
    
    NOISE_MULTIPLIER = args.noise_multiplier
    MAX_GRAD_NORM = args.clipping_bound
    
    model, opt, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM
    )

    # Opacus dataloder process will randomly generate empty batch when batch size is small.
    # Filter out those empty batch from training set.
    trainloader_len = 0
    for batch in train_loader:
        if isinstance(batch, list) and all(tensor.nelement() == 0 for tensor in batch):
                continue
        trainloader_len += 1

    num_training_steps = args.num_epochs * trainloader_len
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    DELTA = 1/trainloader_len
    print(f"Using Sigma = {opt.noise_multiplier:.4f} | C = {opt.max_grad_norm} | Initial DP (ε, δ) = ({privacy_engine.get_epsilon(DELTA)}, {DELTA})")

    progress_bar = tqdm(range(num_training_steps))
    n_steps = 0
    train_loss = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            
            if isinstance(batch, list) and all(tensor.nelement() == 0 for tensor in batch):
                continue 
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_metric.add_batch(predictions=predictions, references=batch['labels'])
            f_train_metric.add_batch(predictions=predictions, references=batch['labels'])
            
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            progress_bar.update(1)

            n_steps += 1
            if n_steps % args.save_every == 0:
                print(f"Using Sigma = {opt.noise_multiplier:.4f} | C = {opt.max_grad_norm} | DP (ε, δ) = ({privacy_engine.get_epsilon(DELTA)}, {DELTA})")
                print('MCC Score: ', train_metric.compute())
                print('F1 Score: ', f_train_metric.compute())
                print('loss train: ', train_loss/n_steps)
                train_loss = 0.0

    model.eval()
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
        f_metric.add_batch(predictions=predictions, references=batch['labels'])

    # Print metrics on validation set.
    print(f"Using Sigma for Validation Set = {opt.noise_multiplier:.4f} | C = {opt.max_grad_norm}")
    print('MCC Score Validation Set: ', metric.compute())
    print('F1 Score Validation Set: ', f_metric.compute())

if __name__ == '__main__':
    main()

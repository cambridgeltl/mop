import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .abstract_processor import convert_examples_to_features
from .common import timeit


class BertTrainer(object):
    def __init__(self, model, optimizer, processor, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.tokenizer = tokenizer
        self.total_step = 0
        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_f1, self.unimproved_iters = 0, 0
        self.early_stop = False

    @timeit
    def train_epoch(self, train_dataloader):
        self.tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.total_step += 1
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            if "roberta" in self.tokenizer.name_or_path:
                input_ids, input_mask, label_ids = batch
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "labels": label_ids,
                }
            else:
                input_ids, input_mask, segment_ids, label_ids = batch
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": input_mask,
                    "labels": label_ids,
                    "token_type_ids": segment_ids,
                }

            if self.args.amp:
                with autocast():
                    if self.args.adapter_names:
                        inputs["adapter_names"] = self.args.adapter_names
                    logits = self.model(**inputs)[0]
            else:
                if self.args.adapter_names:
                    inputs["adapter_names"] = self.args.adapter_names
                logits = self.model(**inputs)[0]

            if self.args.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, label_ids.float())
            else:
                loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()
            if (self.total_step % self.args.save_step == 0) and (
                self.total_step < 20000
            ):
                self.save_model(step=self.total_step)
            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1
        print(self.tr_loss)

    def train_subgraph_cache_tokens(self, group_idx):
        tokenized_features = self.processor.load_and_cache_tokenized_features(
            group_idx, self.tokenizer, self.args
        )
        self.num_train_optimization_steps = (
            int(
                len(tokenized_features)
                / self.args.batch_size
                / self.args.gradient_accumulation_steps
            )
            * self.args.epochs
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_optimization_steps,
        )
        print(f"Start Training on group_idx {group_idx}")
        self.train(tokenized_features, group_idx)

    def train_subgraph(self, group_idx):
        def collate_fn_batch_encoding(batch):
            batch_pair_text = [(example.text_a, example.text_b) for example in batch]
            text_features = self.tokenizer.batch_encode_plus(
                batch_pair_text,
                padding="max_length",  # First sentence will have some PADDED tokens to match second sequence length
                max_length=self.args.max_seq_length,
                return_tensors="pt",
                truncation=True,
            )
            labels = []
            for example in batch:
                n_cls, idx = example.label
                label = np.zeros(n_cls)
                label[idx] = 1
                labels.append(label)
            label_ids = torch.as_tensor(labels, dtype=torch.long)
            if "roberta" in self.args.tokenizer.lower():
                return (
                    text_features.input_ids,
                    text_features.attention_mask,
                    label_ids,
                )
            return (
                text_features.input_ids,
                text_features.attention_mask,
                text_features.token_type_ids,
                label_ids,
            )

        examples = self.processor._create_examples(group_idx)
        self.num_train_optimization_steps = (
            int(
                len(examples)
                / self.args.batch_size
                / self.args.gradient_accumulation_steps
            )
            * self.args.epochs
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_optimization_steps,
        )
        train_dataloader = DataLoader(
            examples,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn_batch_encoding,
        )

        print(f"Start Training on group_idx {group_idx}")
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            self.train_epoch(train_dataloader)
            self.save_model(epoch=epoch, group_idx=group_idx)

    def train(self, tokenized_features, group_idx):
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            train_sampler = RandomSampler(tokenized_features)
            train_dataloader = DataLoader(
                tokenized_features,
                sampler=train_sampler,
                batch_size=self.args.batch_size,
            )
            self.train_epoch(train_dataloader)
            self.save_model(epoch=epoch, group_idx=group_idx)

    def save_model(self, epoch=None, step=None, group_idx=None):
        if epoch is not None and group_idx is not None:
            save_path = f"{self.args.save_path}/group_{group_idx}_epoch_{epoch}/"
        elif step is not None:
            save_path = f"{self.args.save_path}/step_{step}/"

        print(f"saving model to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        if self.args.n_gpu > 1:
            # self.model.module.save_pretrained(save_path)
            self.model.module.save_adapter(save_path, self.args.adapter_names)
        else:
            # self.model.save_pretrained(save_path)
            self.model.save_adapter(save_path, self.args.adapter_names)

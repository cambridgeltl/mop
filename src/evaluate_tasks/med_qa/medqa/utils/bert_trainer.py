import datetime
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

from .abstract_processor import convert_examples_to_features
from .bert_evaluator import BertEvaluator


class BertTrainer(object):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples()
        if args.train_ratio < 1:
            keep_num = int(len(self.train_examples) * args.train_ratio) + 1
            self.train_examples = self.train_examples[:keep_num]
            print(f"Reduce Training example number to {keep_num}")
        model_str = self.args.model
        if "/" in model_str:
            model_str = model_str.split("/")[1]

        self.num_train_optimization_steps = (
            int(
                len(self.train_examples)
                / args.batch_size
                / args.gradient_accumulation_steps
            )
            * args.epochs
        )

        self.log_header = (
            "Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss"
        )
        self.log_template = " ".join(
            "{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}".split(
                ","
            )
        )

        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_acc, self.unimproved_iters = 0, 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        self.tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = self.model(
                input_ids, input_mask, segment_ids, labels=label_ids, return_dict=True
            )
            loss = outputs.loss

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

        print(self.tr_loss)

    def train(self):
        features = convert_examples_to_features(
            self.train_examples, self.tokenizer, self.args
        )
        print("Number of training examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        padded_input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
        padded_input_mask = torch.tensor(features["attention_mask"], dtype=torch.long)
        padded_segment_ids = torch.tensor(features["token_type_ids"], dtype=torch.long)
        label_ids = torch.tensor(features["labels"], dtype=torch.long)

        train_data = TensorDataset(
            padded_input_ids, padded_input_mask, padded_segment_ids, label_ids
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.args.batch_size
        )

        print("Start Training")
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            self.train_epoch(train_dataloader)
            dev_evaluator = BertEvaluator(
                self.model, self.processor, self.tokenizer, self.args, split="dev"
            )
            result = dev_evaluator.get_scores()[0]

            # Update validation results
            if result["accuracy"] > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = result["accuracy"]
                torch.save(self.model, self.args.best_model_dir + "model.bin")
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write(
                        "Early Stopping. Epoch: {}, Best Dev performance: {}".format(
                            epoch, self.best_dev_acc
                        )
                    )
                    break

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
        self.train_examples = self.processor.get_train_examples(
            args.data_dir, args.train_file, args.train_ratio
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        self.best_dev_result, self.unimproved_iters = 0, 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        self.tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = self.model(
                input_ids,
                input_mask,
                segment_ids,
            )
            pooled_output = outputs[1]
            if isinstance(self.model, torch.nn.DataParallel):
                pooled_output = self.model.module.dropout(pooled_output)
                logits = self.model.module.classifier(pooled_output)
            else:
                pooled_output = self.model.dropout(pooled_output)
                logits = self.model.classifier(pooled_output)

            loss = F.binary_cross_entropy_with_logits(logits, label_ids.float())

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
        train_features = convert_examples_to_features(
            self.train_examples, self.args.max_seq_length, self.tokenizer
        )
        unpadded_input_ids = [f.input_ids for f in train_features]
        unpadded_input_mask = [f.input_mask for f in train_features]
        unpadded_segment_ids = [f.segment_ids for f in train_features]

        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

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
            if result["micro_f1"] > self.best_dev_result:
                self.unimproved_iters = 0
                self.best_dev_result = result["micro_f1"]
                torch.save(self.model, self.args.best_model_dir + "model.bin")
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write(
                        "Early Stopping. Epoch: {}, Best Dev performance: {}".format(
                            epoch, self.best_dev_result
                        )
                    )
                    break

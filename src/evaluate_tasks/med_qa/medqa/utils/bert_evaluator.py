import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from tabulate import tabulate
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from utils.preprocessing import pad_input_matrix

from .abstract_processor import (
    convert_examples_to_features,
    convert_examples_to_features_long,
    convert_examples_to_hierarchical_features,
)

# Suppress warnings from sklearn.metrics
warnings.filterwarnings("ignore")


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table.
    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.
    Returns:
        None
    """
    print("-" * 80)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 80)
    return tabulate(df, headers=columns, tablefmt="psql")


class BertEvaluator(object):
    def __init__(
        self, model, processor, tokenizer, args, split="dev", dump_predictions=False
    ):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.dump_predictions = dump_predictions

        if split == "train":
            self.eval_examples = self.processor.get_train_examples()
            if args.train_ratio < 1:
                keep_num = int(len(self.eval_examples) * args.train_ratio) + 1
                self.eval_examples = self.eval_examples[:keep_num]
                print(f"Reduce Training example number to {keep_num}")
        elif split == "dev":
            self.eval_examples = self.processor.get_dev_examples()
        elif split == "test":
            self.eval_examples = self.processor.get_test_examples()
        self.examples_ids = [example["id"] for example in self.eval_examples]

    def get_scores(self, silent=False):
        print("Number of evaluation examples: ", len(self.eval_examples))
        features = convert_examples_to_features(
            self.eval_examples, self.tokenizer, self.args
        )
        padded_input_ids = torch.tensor(features["input_ids"], dtype=torch.long)
        padded_input_mask = torch.tensor(features["attention_mask"], dtype=torch.long)
        padded_segment_ids = torch.tensor(features["token_type_ids"], dtype=torch.long)
        label_ids = torch.tensor(features["labels"], dtype=torch.long)

        eval_data = TensorDataset(
            padded_input_ids, padded_input_mask, padded_segment_ids, label_ids
        )

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.args.batch_size
        )
        self.model.eval()

        total_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels, target_labels = list(), list()

        for input_ids, input_mask, segment_ids, label_ids in tqdm(
            eval_dataloader, desc="Evaluating", disable=silent
        ):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            label_ids = label_ids.to(self.args.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    input_mask,
                    segment_ids,
                    labels=label_ids,
                    return_dict=True,
                )
                predicts = outputs.logits.argmax(dim=-1)
            loss = outputs.loss
            predicted_labels.extend(predicts.cpu().detach().numpy())
            target_labels.extend(label_ids.cpu().detach().numpy())

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            total_loss += loss.item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        predicted_labels_dict = {}
        for idx, example_id in enumerate(self.examples_ids):
            predicted_labels_dict[example_id] = predicted_labels[idx]
        predicted_labels, target_labels = (
            np.array(predicted_labels),
            np.array(target_labels),
        )
        model_str = self.args.model
        if "/" in model_str:
            model_str = model_str.split("/")[1]
        avg_loss = total_loss / nb_eval_steps
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        result = {
            "accuracy": accuracy,
            "avg_loss": avg_loss,
        }
        print(metrics.classification_report(target_labels, predicted_labels))
        return (
            result,
            predicted_labels_dict,
        )

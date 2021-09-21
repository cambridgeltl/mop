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


def hoc_label_combine(predicted_labels, target_labels, sentences_ids):
    data = {}
    assert len(predicted_labels) == len(target_labels)
    assert len(predicted_labels) == len(sentences_ids)
    for idx, key_id in enumerate(sentences_ids):
        key = key_id[: key_id.find("_")]
        if key not in data:
            data[key] = (np.zeros(10), np.zeros(10))
        data[key][0][np.nonzero(predicted_labels[idx])] = 1
        data[key][1][np.nonzero(target_labels[idx])] = 1
    predicted_labels = []
    target_labels = []
    for _, doc in data.items():
        predicted_labels.append(doc[0])
        target_labels.append(doc[1])
    print(
        f"Collect labels for {len(predicted_labels)} document from {len(sentences_ids)} sentences "
    )
    return np.array(predicted_labels), np.array(target_labels)


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

        if split == "test":
            self.eval_examples = self.processor.get_test_examples()
        elif split == "dev":
            self.eval_examples = self.processor.get_dev_examples()
        else:
            self.eval_examples = self.processor.get_train_examples()
            if args.train_ratio < 1:
                keep_num = int(len(self.eval_examples) * args.train_ratio) + 1
                print(f"Reduce Training example number to {keep_num}")
                self.eval_examples = self.eval_examples[:keep_num]
        self.examples_ids = [example.guid for example in self.eval_examples]

    def get_scores(self, silent=False):
        eval_features = convert_examples_to_features(
            self.eval_examples, self.args.max_seq_length, self.tokenizer
        )

        unpadded_input_ids = [f.input_ids for f in eval_features]
        unpadded_input_mask = [f.input_mask for f in eval_features]
        unpadded_segment_ids = [f.segment_ids for f in eval_features]

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

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
                if (
                    self.args.n_gpu > 0 and input_ids.size()[0] % self.args.n_gpu != 0
                ):  # For fixing the bug of sample size less than the gpu size.
                    print(
                        "input_ids.size()[0] < self.args.n_gpu ---> back to single gpu mode"
                    )
                    self.model = self.model.module
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
                if (
                    self.args.n_gpu > 0 and input_ids.size()[0] % self.args.n_gpu != 0
                ):  # For fixing the bug of sample size less than the gpu size.
                    self.model = torch.nn.DataParallel(self.model)

            predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
            target_labels.extend(torch.argmax(label_ids, dim=1).cpu().detach().numpy())
            loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

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
        if self.args.num_labels == 2:
            precision = metrics.precision_score(
                target_labels,
                predicted_labels,
            )
            recall = metrics.recall_score(
                target_labels,
                predicted_labels,
            )
            f1 = metrics.f1_score(target_labels, predicted_labels, average="micro")
            binary_result = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            result.update(binary_result)
        else:
            macro_precision = metrics.precision_score(
                target_labels, predicted_labels, average="macro"
            )
            macro_recall = metrics.recall_score(
                target_labels, predicted_labels, average="macro"
            )
            macro_f1 = metrics.f1_score(
                target_labels, predicted_labels, average="macro"
            )
            weighted_precision = metrics.precision_score(
                target_labels, predicted_labels, average="weighted"
            )
            weighted_recall = metrics.recall_score(
                target_labels, predicted_labels, average="weighted"
            )
            weighted_f1 = metrics.f1_score(
                target_labels, predicted_labels, average="weighted"
            )
            muticlass_result = {
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
            }
            result.update(muticlass_result)

        return (
            result,
            predicted_labels_dict,
        )

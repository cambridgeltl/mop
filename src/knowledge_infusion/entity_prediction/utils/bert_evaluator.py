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

from utils.abstract_processor import convert_examples_to_features

# Suppress warnings from sklearn.metrics
warnings.filterwarnings("ignore")


def pad_input_matrix(unpadded_matrix, max_doc_length):
    """
    Returns a zero-padded matrix for a given jagged list
    :param unpadded_matrix: jagged list to be padded
    :return: zero-padded matrix
    """
    max_doc_length = min(max_doc_length, max(len(x) for x in unpadded_matrix))
    zero_padding_array = [0 for i0 in range(len(unpadded_matrix[0][0]))]

    for i0 in range(len(unpadded_matrix)):
        if len(unpadded_matrix[i0]) < max_doc_length:
            unpadded_matrix[i0] += [
                zero_padding_array
                for i1 in range(max_doc_length - len(unpadded_matrix[i0]))
            ]
        elif len(unpadded_matrix[i0]) > max_doc_length:
            unpadded_matrix[i0] = unpadded_matrix[i0][:max_doc_length]


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
            self.eval_examples = self.processor.get_test_examples(
                args.data_dir, args.test_file
            )
        else:
            self.eval_examples = self.processor.get_dev_examples(
                args.data_dir, args.dev_file
            )
        self.examples_ids = [example.guid for example in self.eval_examples]

    def get_scores(self, silent=False):
        if self.args.is_hierarchical:
            eval_features = convert_examples_to_hierarchical_features(
                self.eval_examples, self.args.max_seq_length, self.tokenizer
            )
        else:
            if "longformer" in self.args.model:
                eval_features = convert_examples_to_features_long(
                    self.eval_examples, self.args.max_seq_length, self.tokenizer
                )
            elif "reformer" in self.args.model:
                eval_features = convert_examples_to_features_long(
                    self.eval_examples,
                    self.args.max_seq_length,
                    self.tokenizer,
                    "reformer",
                )
            else:
                eval_features = convert_examples_to_features(
                    self.eval_examples, self.args.max_seq_length, self.tokenizer
                )

        unpadded_input_ids = [f.input_ids for f in eval_features]
        unpadded_input_mask = [f.input_mask for f in eval_features]
        unpadded_segment_ids = [f.segment_ids for f in eval_features]

        if self.args.is_hierarchical:
            pad_input_matrix(unpadded_input_ids, self.args.max_doc_length)
            pad_input_matrix(unpadded_input_mask, self.args.max_doc_length)
            pad_input_matrix(unpadded_segment_ids, self.args.max_doc_length)

        padded_input_ids = torch.as_tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.as_tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.as_tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids = torch.as_tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )

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
                    input_ids.size()[0] < self.args.n_gpu
                ):  # For fixing the bug of sample size less than the gpu size.
                    print(
                        "input_ids.size()[0] < self.args.n_gpu ---> back to single gpu mode"
                    )
                    self.model = self.model.module
                logits = self.model(
                    input_ids,
                    input_mask,
                    segment_ids,
                    adapter_names=self.args.adapter_names,
                )[0]
                if input_ids.size()[0] < self.args.n_gpu:
                    self.model = torch.nn.DataParallel(self.model)

            if self.args.is_multilabel:
                predicted_labels.extend(
                    F.sigmoid(logits).round().long().cpu().detach().numpy()
                )
                target_labels.extend(label_ids.cpu().detach().numpy())
                loss = F.binary_cross_entropy_with_logits(
                    logits, label_ids.float(), size_average=False
                )
            else:
                predicted_labels.extend(
                    torch.argmax(logits, dim=1).cpu().detach().numpy()
                )
                target_labels.extend(
                    torch.argmax(label_ids, dim=1).cpu().detach().numpy()
                )
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

        if self.processor.NAME == "HOC" or self.processor.NAME == "Hoc":
            predicted_labels, target_labels = hoc_label_combine(
                predicted_labels, target_labels, self.examples_ids
            )

        model_str = self.args.model
        if "/" in model_str:
            model_str = model_str.split("/")[1]
        if self.dump_predictions:
            pickle.dump(
                (predicted_labels, target_labels),
                open(
                    os.path.join(
                        self.args.data_dir,
                        self.args.dataset,
                        "{}_{}_{}_{}_predictions.p".format(
                            self.split,
                            model_str,
                            self.args.training_file,
                            self.args.max_seq_length,
                        ),
                    ),
                    "wb",
                ),
            )
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

        if self.args.is_multilabel:
            micro_precision = metrics.precision_score(
                target_labels, predicted_labels, average="micro"
            )
            micro_recall = metrics.recall_score(
                target_labels, predicted_labels, average="micro"
            )
            micro_f1 = metrics.f1_score(
                target_labels, predicted_labels, average="micro"
            )
            mutilabel_result = {
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }
            result.update(mutilabel_result)
        return (
            result,
            predicted_labels_dict,
        )

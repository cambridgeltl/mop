"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import csv
import sys

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeaturesText(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets a list of possible labels in the dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        import sys

        csv.field_size_limit(sys.maxsize)

        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


def convert_examples_to_features(examples, tokenizer, args):
    # Preprocessing the datasets.
    option_name = "options"
    context_name = "contexts"
    question_name = "question"
    label_name = "answer_idx"
    num_options = 5
    top_k_context = 20

    first_sentences = []
    second_sentences = []
    labels = []
    for example in examples:
        first_sentences.append(
            [
                "".join(example[context_name][index][:top_k_context])
                for index in range(num_options)
            ]
        )
        question_sentence = example[question_name]
        second_sentences.append(
            [f"{question_sentence} {option}" for option in example[option_name]]
        )
        labels.append(example[label_name])

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=args.max_seq_length,
        padding="max_length",
    )
    inputs = {
        k: [v[i : i + num_options] for i in range(0, len(v), num_options)]
        for k, v in tokenized_examples.items()
    }
    inputs["labels"] = labels
    # input_ids_li = inputs["input_ids"]
    # token_type_ids_li = inputs["token_type_ids"]
    # attention_mask_li = inputs["attention_mask"]
    # features = []
    # for idx, label_id in enumerate(labels):
    #     features.append(
    #         InputFeatures(
    #             input_ids=input_ids_li[idx],
    #             input_mask=attention_mask_li[idx],
    #             segment_ids=token_type_ids_li[idx],
    #             label_id=label_id,
    #         )
    #     )
    return inputs


def convert_examples_to_features_long(
    examples, max_seq_length, tokenizer, print_examples=False, model_type="longformer"
):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []

    encoded_out = tokenizer.batch_encode_plus(
        [example.text_a.replace("\\n", "</s>") for example in examples],
        add_special_tokens=True,
        max_length=max_seq_length,
        pad_to_max_length=True,
        return_token_type_ids=True,
    )

    input_ids = encoded_out["input_ids"]
    attention_masks = encoded_out["attention_mask"]
    segment_ids = encoded_out["token_type_ids"]

    for example, ids, masks, segments in zip(
        examples, input_ids, attention_masks, segment_ids
    ):

        if model_type == "longformer":
            masks[0] = 2

        label_id = [float(x) for x in example.label]

        features.append(
            InputFeatures(
                input_ids=ids, input_mask=masks, segment_ids=segments, label_id=label_id
            )
        )
    return features


def convert_examples_to_hierarchical_features(
    examples, max_seq_length, tokenizer, print_examples=False
):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_a)]
        tokens_b = None

        if example.text_b:
            tokens_b = [
                tokenizer.tokenize(line) for line in sent_tokenize(example.text_b)
            ]
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length
            # Account for [CLS], [SEP], [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP]
            for i0 in range(len(tokens_a)):
                if len(tokens_a[i0]) > max_seq_length - 2:
                    tokens_a[i0] = tokens_a[i0][: (max_seq_length - 2)]

        tokens = [["[CLS]"] + line + ["[SEP]"] for line in tokens_a]
        segment_ids = [[0] * len(line) for line in tokens]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = list()
        for line in tokens:
            input_ids.append(tokenizer.convert_tokens_to_ids(line))

        # Input mask has 1 for real tokens and 0 for padding tokens
        input_mask = [[1] * len(line_ids) for line_ids in input_ids]

        # Zero-pad up to the sequence length.
        padding = [[0] * (max_seq_length - len(line_ids)) for line_ids in input_ids]
        for i0 in range(len(input_ids)):
            input_ids[i0] += padding[i0]
            input_mask[i0] += padding[i0]
            segment_ids[i0] += padding[i0]

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

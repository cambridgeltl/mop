import csv
import sys

from .common import timeit


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


@timeit
def convert_examples_to_features(
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

    batch_pair_text = [(example.text_a, example.text_b) for example in examples]
    labels = [[float(x) for x in example.label] for example in examples]
    text_features = tokenizer.batch_encode_plus(
        batch_pair_text,
        padding="max_length",  # First sentence will have some PADDED tokens to match second sequence length
        max_length=max_seq_length,
        return_tensors="pt",
        truncation=True,
    )
    return text_features, labels

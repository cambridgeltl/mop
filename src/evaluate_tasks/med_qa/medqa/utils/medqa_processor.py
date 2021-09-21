import json
import os

from .abstract_processor import BertProcessor, InputExample


def load_dataset_from_file(data_dir):
    train_file = "train_context.jsonl"
    dev_file = "dev_context.jsonl"
    test_file = "test_context.jsonl"
    train_file = os.path.join(data_dir, train_file)
    dev_file = os.path.join(data_dir, dev_file)
    test_file = os.path.join(data_dir, test_file)
    train_data_li = []
    with open(train_file) as f:
        for idx, line in enumerate(f.readlines()):
            data = json.loads(line)
            data["id"] = idx
            train_data_li.append(data)

    dev_data_li = []
    with open(dev_file) as f:
        for idx, line in enumerate(f.readlines()):
            data = json.loads(line)
            data["id"] = idx
            dev_data_li.append(data)

    test_data_li = []
    with open(test_file) as f:
        for idx, line in enumerate(f.readlines()):
            data = json.loads(line)
            data["id"] = idx
            test_data_li.append(data)

    return {
        "train": train_data_li,
        "dev": dev_data_li,
        "test": test_data_li,
    }


class MedQaProcessor(BertProcessor):
    NAME = "MEDQA"

    def __init__(self, data_dir):
        self.datasets = load_dataset_from_file(data_dir)

    def get_train_examples(self):
        return self.datasets["train"]

    def get_dev_examples(self):
        return self.datasets["dev"]

    def get_test_examples(self):
        return self.datasets["test"]

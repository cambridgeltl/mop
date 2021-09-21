import pandas as pd

from .abstract_processor import BertProcessor, InputExample


def load_bioasq(file_path):
    train_df = pd.read_csv(
        f"{file_path}train.tsv",
        sep="\t",
        header=0,
        names=["id", "text_a", "text_b", "label"],
    )
    print(train_df)
    dev_df = pd.read_csv(
        f"{file_path}dev.tsv",
        sep="\t",
        header=0,
        names=["id", "text_a", "text_b", "label"],
    )
    test_df = pd.read_csv(
        f"{file_path}test.tsv",
        sep="\t",
        header=0,
        names=["id", "text_a", "text_b", "label"],
    )
    return train_df, dev_df, test_df


class NLIProcessor(BertProcessor):
    NAME = "MED_NLI"
    NUM_CLASSES = 3
    IS_MULTILABEL = False

    def __init__(self, data_dir):
        self.train_df, self.dev_df, self.test_df = load_bioasq(data_dir)

    def get_train_examples(self, data_dir, filename="train.tsv"):
        return self._create_examples(self.train_df, set_type="train")

    def get_dev_examples(self, data_dir, filename="dev.tsv"):
        return self._create_examples(self.dev_df, set_type="dev")

    def get_test_examples(self, data_dir, filename="test.tsv"):
        return self._create_examples(self.test_df, set_type="test")

    def _create_examples(self, data_df, set_type):
        examples = []
        label_map = {
            "entailment": "001",
            "neutral": "010",
            "contradiction": "100",
        }
        for (i, row) in data_df.iterrows():
            guid = row["id"]
            text_a = row["text_a"]
            text_b = row["text_b"]
            label = label_map[row["label"]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets for {set_type} set"
        )
        return examples

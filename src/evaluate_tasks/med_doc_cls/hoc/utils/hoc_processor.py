import os

from .abstract_processor import BertProcessor, InputExample


class HOCProcessor(BertProcessor):
    NAME = "Hoc"
    NUM_CLASSES = 10
    IS_MULTILABEL = True

    def _read_tsv(self, input_file):
        print(f"Reading data from: {input_file}")
        with open(input_file, "r") as f:
            f.readline()
            return f.readlines()

    def get_train_examples(self, data_dir, filename="train.tsv", train_ratio=1):
        if train_ratio != 1:
            filename = f"train_{train_ratio}.tsv"
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, filename)), "train"
        )

    def get_dev_examples(self, data_dir, filename="dev.tsv"):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, filename)), "dev"
        )

    def get_test_examples(self, data_dir, filename="test.tsv"):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, filename)), "test"
        )

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # guid = '%s-%s' % (set_type, i)
            line = line.strip().split("\t")
            text_a = line[1]
            label = [float(l.split("_")[1]) for l in line[0].split(",")]
            guid = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets for {set_type} set"
        )
        return examples

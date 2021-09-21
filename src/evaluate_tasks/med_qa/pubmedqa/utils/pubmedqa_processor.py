import json

import pandas as pd

from .abstract_processor import BertProcessor, InputExample


def load_pubmedqa(data_dir, fold_num=0):
    train_json = json.load(open(f"{data_dir}/pqal_fold{fold_num}/train_set.json"))
    dev_json = json.load(open(f"{data_dir}/pqal_fold{fold_num}/dev_set.json"))
    test_json = json.load(open(f"{data_dir}/test_set.json"))

    id_li = []
    question_li = []
    context_li = []
    label_li = []
    for k, v in train_json.items():
        id_li.append(k)
        question_li.append(v["QUESTION"])
        context_li.append(v["CONTEXTS"])
        label_li.append(v["final_decision"])
    train_df = pd.DataFrame(
        {"id": id_li, "question": question_li, "context": context_li, "label": label_li}
    )

    dev_id_li = []
    dev_question_li = []
    dev_context_li = []
    dev_label_li = []
    for k, v in dev_json.items():
        dev_id_li.append(k)
        dev_question_li.append(v["QUESTION"])
        dev_context_li.append(v["CONTEXTS"])
        dev_label_li.append(v["final_decision"])
    dev_df = pd.DataFrame(
        {
            "id": dev_id_li,
            "question": dev_question_li,
            "context": dev_context_li,
            "label": dev_label_li,
        }
    )

    test_id_li = []
    test_question_li = []
    test_context_li = []
    test_label_li = []
    for k, v in test_json.items():
        test_id_li.append(k)
        test_question_li.append(v["QUESTION"])
        test_context_li.append(v["CONTEXTS"])
        test_label_li.append(v["final_decision"])
    test_df = pd.DataFrame(
        {
            "id": test_id_li,
            "question": test_question_li,
            "context": test_context_li,
            "label": test_label_li,
        }
    )
    print(
        f"Load pubmed_qa_l datasets train_df({len(train_df.index)}),dev_df({len(dev_df.index)}),test_df({len(test_df.index)})"
    )
    return train_df, dev_df, test_df


class PubMedQAProcessor(BertProcessor):
    NAME = "PubMedQA"
    NUM_CLASSES = 3
    IS_MULTILABEL = False

    def __init__(self, data_dir, fold_num=0):
        self.train_df, self.dev_df, self.test_df = load_pubmedqa(
            data_dir, fold_num=fold_num
        )

    def get_train_examples(self):
        return self._create_examples(self.train_df, set_type="train")

    def get_dev_examples(self):
        return self._create_examples(self.dev_df, set_type="dev")

    def get_test_examples(self):
        return self._create_examples(self.test_df, set_type="test")

    def _create_examples(self, data_df, set_type):
        examples = []
        label_map = {
            "yes": "001",
            "no": "010",
            "maybe": "100",
        }
        for (i, row) in data_df.iterrows():
            guid = row["id"]
            text_a = row["question"]
            text_b = " ".join(row["context"])
            label = label_map[row["label"]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets for {set_type} set"
        )
        return examples

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from utils.abstract_processor import (
    BertProcessor,
    InputExample,
    convert_examples_to_features,
)
from utils.common import _construct_adj, partition_graph, timeit


class KGProcessor(BertProcessor):
    def __init__(
        self,
        data_dir,
        sub_set=1,
        name="Node Prediction With Partition",
        n_partition=50,
        bi_direction=True,
        sub_group_idx=None,
        shuffle_rate=None,
    ):
        self.NAME = name
        self.id2ent = {}
        self.id2rel = {}
        self.n_partition = n_partition
        self.sub_set = sub_set
        self.tri_file = os.path.join(data_dir, "train2id.txt")
        self.ent_file = os.path.join(data_dir, "entity2id.txt")
        self.rel_file = os.path.join(data_dir, "relation2id.txt")
        if shuffle_rate:
            self.partition_file = os.path.join(
                data_dir, f"partition_{n_partition}_shuf_{shuffle_rate}.txt"
            )
            print(self.partition_file)
            assert os.path.exists(self.partition_file)
        else:
            self.partition_file = os.path.join(data_dir, f"partition_{n_partition}.txt")
        if sub_group_idx is not None:
            self.partition_file = os.path.join(
                data_dir, f"partition_{n_partition}_{sub_group_idx}_{n_partition}.txt"
            )
        self.bi_direction = bi_direction
        self.cache_feature_base_dir = os.path.join(
            data_dir, "feature_cache_metis_partition/"
        )
        os.makedirs(self.cache_feature_base_dir, exist_ok=True)
        self.examples_cache = {}  # Memory cache of loaded partion nodes
        self.load_data(sub_set)
        ## Wether to prediction both the head and tail nodes
        ## if bi_direction is False, only predict the tail node
        super(KGProcessor, self).__init__()

    def partition_graph(self, partition_n):
        print("Start partition_graph")
        h_list = []
        t_list = []
        r_list = []
        with open(self.tri_file, "r") as f:
            print(f"loading triples {f.readline()}")
            for line in tqdm(f.readlines()):
                h, t, r = line.split("\t")
                h_list.append(int(h.strip()))
                t_list.append(int(t.strip()))
                r_list.append(int(r.strip()))
        triple_df = pd.DataFrame(
            {
                "head_id": h_list,
                "relation_id": r_list,
                "tail_id": t_list,
            }
        )
        edge_list = []
        for i, row in triple_df.iterrows():
            edge_list.append([row.head_id, row.tail_id])
        edge_list_ar = np.array(edge_list)
        num_nodes = self.ent_total
        adj = _construct_adj(edge_list_ar, num_nodes)
        idx_nodes = [i for i in range(self.ent_total)]
        part_adj, parts = partition_graph(adj, idx_nodes, partition_n)
        with open(self.partition_file, "w") as f:
            for node_list in parts:
                f.write("\t".join([str(i) for i in node_list]) + "\n")

    def load_data(self, sub_set):

        ## Read entity file
        with open(self.ent_file, "r") as f:
            self.ent_total = (int)(f.readline())
            if sub_set != 1:
                self.ent_total = int(self.ent_total * sub_set)

            for ent in f.readlines():
                if sub_set != 1:
                    if int(ent.split("\t")[1]) < self.ent_total:
                        self.id2ent[int(ent.split("\t")[1].strip())] = ent.split("\t")[
                            0
                        ]
                else:
                    self.id2ent[int(ent.split("\t")[1].strip())] = ent.split("\t")[0]
            print(
                f"Loading entities (subset mode:{sub_set}) ent_total:{self.ent_total} len(self.id2ent): {len(self.id2ent)}"
            )

        ## Read Relation File
        with open(self.rel_file, "r") as f:
            print("Read Relation File")
            self.rel_total = (int)(f.readline())  # num of total relations
            for rel in f.readlines():
                self.id2rel[int(rel.split("\t")[1].strip())] = rel.split("\t")[0]
            print(f"{len(self.id2rel)} relations loaded.")

        ##  Read Partition File
        self.node_group_idx = {}  # A dict saving the group index of each entity
        self.num_class_list = []  # A list saving the number of entities for each group
        self.nodes_partition = []  # A list of node dicts

        if not os.path.exists(self.partition_file):
            # Do partitioning if the partitioned file not exist.
            self.partition_graph(self.n_partition)

        with open(self.partition_file, "r") as f:
            print(f"Reading partition file: {self.partition_file}.")
            for group_idx, line in enumerate(f.readlines()):
                nodes = {
                    int(eid.strip()): idx for idx, eid in enumerate(line.split("\t"))
                }
                for eid in line.split("\t"):
                    self.node_group_idx[int(eid.strip())] = group_idx
                self.nodes_partition.append(nodes)
                self.num_class_list.append(len(nodes))

            print(f"{len(self.nodes_partition)} partitioned groups loaded. ")
            print(
                f"Number for nodes in each partitions: min({min(self.num_class_list)}),max({max(self.num_class_list)})"
            )
            print(
                f"Total Nodes number: {self.ent_total}, Nodes number in partitions:{len(self.node_group_idx)}"
            )
        ## Read Triple File
        f = open(self.tri_file, "r")
        triples_total = (int)(f.readline())

        count = 0
        self.triple_list = [[] for i in range(self.n_partition)]
        for line in f.readlines():
            h, t, r = line.strip().split("\t")
            if (
                ((int)(h) in self.id2ent)
                and ((int)(t) in self.id2ent)
                and ((int)(r) in self.id2rel)
                and ((int)(h) in self.node_group_idx)
                and ((int)(t) in self.node_group_idx)
            ):
                group_idx_h = self.node_group_idx[(int)(h)]
                group_idx_t = self.node_group_idx[(int)(t)]
                if group_idx_h == group_idx_t:
                    self.triple_list[group_idx_t].append(((int)(h), (int)(t), (int)(r)))
                    count += 1
        f.close()
        if triples_total != count:
            print(
                f"Using sub-set mode or some triples are missing, total:{triples_total} --> subset:{count}"
            )
            triples_total = count

    @timeit
    def _create_examples(self, group_idx: int):
        if group_idx in self.examples_cache:
            print(
                f"Get cache examples from partition {group_idx}/{self.n_partition} set"
            )
            return self.examples_cache[group_idx]
        examples = []

        def cls_one_hot(ent_id):
            """[Get the onehot_tuple (num_class, idx of entity in the class)]

            Args:
                ent_id ([type]): [description]

            Returns:
                [type]: [description]
            """
            return (
                self.num_class_list[group_idx],
                self.nodes_partition[group_idx][ent_id],
            )

        for (h_id, t_id, r_id) in self.triple_list[group_idx]:
            text_h = self.id2ent[h_id]
            text_t = self.id2ent[t_id]
            text_r = self.id2rel[r_id]
            if self.bi_direction:
                examples.append(  # use text_h+test_r to predict t_id
                    InputExample(
                        guid=None,
                        text_a=text_h,
                        text_b=text_r,
                        label=cls_one_hot(t_id),
                    )
                )
                examples.append(  # use test_r+text_t to predict h_id
                    InputExample(
                        guid=None,
                        text_a=text_r,
                        text_b=text_t,
                        label=cls_one_hot(h_id),
                    )
                )
            else:
                examples.append(  # use text_h+test_r to predict t_id
                    InputExample(
                        guid=None,
                        text_a=text_h,
                        text_b=text_r,
                        label=cls_one_hot(t_id),
                    )
                )
        self.examples_cache[group_idx] = examples
        print(
            f"Get {len(examples)} examples of {self.NAME} datasets from partition {group_idx}/{self.n_partition} set"
        )
        return examples

    @timeit
    def load_and_cache_tokenized_features(self, group_idx: int, tokenizer, args):
        bi_direct_str = "_bi_dir" if self.bi_direction else ""
        cached_features_file = os.path.join(
            self.cache_feature_base_dir,
            f"feature_{self.n_partition}_{group_idx}{bi_direct_str}.pt",
        )
        if os.path.exists(cached_features_file):
            tokenized_features = torch.load(cached_features_file)
        else:
            current_example = self._create_examples(group_idx)
            text_features, labels = convert_examples_to_features(
                current_example, args.max_seq_length, tokenizer
            )
            label_ids = torch.as_tensor(labels, dtype=torch.long)

            tokenized_features = TensorDataset(
                text_features.input_ids,
                text_features.attention_mask,
                text_features.token_type_ids,
                label_ids,
            )
            torch.save(tokenized_features, cached_features_file)
        return tokenized_features

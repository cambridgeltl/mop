import os
import random
import shutil
import time
from datetime import datetime
from os import listdir
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch
from transformers import (
    AdamW,
    AdapterConfig,
    AdapterFusionConfig,
    AdapterType,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

import wandb
from args import get_args
from utils.bert_evaluator import BertEvaluator
from utils.bert_trainer import BertTrainer
from utils.common_utils import print_args_as_table
from utils.pubmedqa_processor import PubMedQAProcessor

wandb.init(project="PubMedQA")
sapbert_dir = "/home/fl399/entity-linking-21/src/umls_pretraining_adapters/tmp/pubmed_umls_pretrained_adapters/new_implmentation_self_retreive_full_umls_11792953_maxlen25_start_from_pubmed_bert_adapters_crate_16_bs256_2gpu_random_seed_2049/"
rel_pred_dir = "/home/zm324/workspace/models/kg_bert/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_snomed_ro_top20_20210305_111449_adapter"


def evaluate_split(model, processor, tokenizer, args, split="dev"):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split, True)
    result = evaluator.get_scores(silent=True)
    split_result = {}
    for k, v in result[0].items():
        split_result[f"{split}_{k}"] = v
    return split_result, result[1]


def get_tf_flag(args):
    from_tf = False
    if (
        (
            ("BioRedditBERT" in args.model)
            or ("BioBERT" in args.model)
            or ("SapBERT" in args.model)
        )
        and "step_" not in args.model
        and "epoch_" not in args.model
    ):
        from_tf = True

    if ("SapBERT" in args.model) and ("original" in args.model):
        from_tf = False
    return from_tf


def search_adapters(args):
    """[Search the model_path, take all the sub directions as adapter_names]

    Args:
        args (ArgumentParser)

    Returns:
        [dict]: {model_path:[adapter_names]}
    """
    adapter_paths_dic = {}
    if "," in args.model:
        for model in args.model.split(","):  # need to fusion from two or more models
            model_path = args.model_dir + model
            adapter_paths = [f for f in listdir(model_path)]
            print(f"Found {len(adapter_paths)} adapter paths")
            adapter_paths = check_adapter_names(model_path, adapter_paths)
            adapter_paths_dic[model_path] = adapter_paths
    else:
        model_path = args.model_dir + args.model
        adapter_paths = [f for f in listdir(model_path)]
        print(f"Found {len(adapter_paths)} adapter paths")
        adapter_paths = check_adapter_names(model_path, adapter_paths)
        adapter_paths_dic[model_path] = adapter_paths
    return adapter_paths_dic


def check_adapter_names(model_path, adapter_names):
    """[Check if the adapter path contrains the adapter model]

    Args:
        model_path ([type]): [description]
        adapter_names ([type]): [description]

    Raises:
        ValueError: [description]
    """
    checked_adapter_names = []
    print(f"Checking adapter namer:{model_path}:{len(adapter_names)}")
    for adapter_name in adapter_names:  # group_0_epoch_1
        adapter_model_path = os.path.join(model_path, adapter_name)
        if f"epoch_{args.pretrain_epoch}" not in adapter_name:
            # check pretrain_epoch
            continue
        if args.groups and int(adapter_name.split("_")[1]) not in set(args.groups):
            # check selected groups
            continue
        adapter_model_path = os.path.join(adapter_model_path, "pytorch_adapter.bin")
        assert os.path.exists(
            adapter_model_path
        ), f"{adapter_model_path} adapter not found."

        checked_adapter_names.append(adapter_name)
    print(f"Valid adapters ({len(checked_adapter_names)}):{checked_adapter_names}")
    return checked_adapter_names


def load_fusion_adapter_model(args, adapter_names_dict):
    base_model = AutoModel.from_pretrained(args.base_model)
    fusion_adapter_rename = []
    for model_path, adapter_names in adapter_names_dict.items():
        for adapter_name in adapter_names:
            adapter_dir = os.path.join(model_path, adapter_name)
            new_adapter_name = model_path[-14:][:-8] + "_" + adapter_name
            base_model.load_adapter(adapter_dir, load_as=new_adapter_name)
            print(f"add adapter_name:{new_adapter_name}")
            fusion_adapter_rename.append(new_adapter_name)
    fusion_config = AdapterFusionConfig.load("dynamic", temperature=args.temperature)
    base_model.add_fusion(fusion_adapter_rename, fusion_config)
    # base_model.train_fusion(fusion_adapter_rename)
    base_model.encoder.enable_adapters(fusion_adapter_rename, True, True)
    base_model.set_active_adapters(fusion_adapter_rename)
    config = AutoConfig.from_pretrained(
        os.path.join(adapter_dir, "adapter_config.json")
    )
    # base_model.train_fusion([adapter_names])
    if args.add_sapbert:
        base_model.load_adapter(sapbert_dir, load_as="sapbert")
        adapter_names.append("sapbert")
    if args.add_rel_pred:
        base_model.load_adapter(rel_pred_dir + "/epoch_0/", load_as="rel_pred_0")
        adapter_names.append("rel_pred_0")
        base_model.load_adapter(rel_pred_dir + "/epoch_1/", load_as="rel_pred_1")
        adapter_names.append("rel_pred_1")
    return config, base_model


if __name__ == "__main__":
    args = get_args()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    model_str = args.model
    if "/" in model_str:
        model_str = model_str.split("/")[1]

    print("Device:", str(device).upper())
    print("Number of GPUs:", n_gpu)
    print("AMP:", args.amp)
    if args.groups:
        args.groups = [int(i) for i in args.groups.split(",")]
    print("Groups:", args.groups)
    train_acc_list = []
    train_f1_list = []
    dev_acc_list = []
    dev_f1_list = []
    test_acc_list = []
    test_f1_list = []

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = PubMedQAProcessor.NUM_CLASSES
    args.is_multilabel = PubMedQAProcessor.IS_MULTILABEL

    # Record config on wandb
    wandb.config.update(args)
    print_args_as_table(args)

    if args.tokenizer is None:
        args.tokenizer = args.model

    for i in range(args.repeat_runs):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.best_model_dir = f"./temp/model_{timestamp_str}/"
        os.makedirs(args.best_model_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        adapter_names_dict = search_adapters(args)

        processor = PubMedQAProcessor(args.data_dir, fold_num=i)
        # Set random seed for reproducibility
        seed = int(time.time())
        print(f"Generate random seed {seed}. Start training on fold {i}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        config, model = load_fusion_adapter_model(args, adapter_names_dict)
        model.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        model.classifier = torch.nn.Linear(config.hidden_size, 3)

        train_examples = None
        num_train_optimization_steps = None
        train_examples = processor.get_train_examples()
        num_train_optimization_steps = (
            int(
                len(train_examples) / args.batch_size / args.gradient_accumulation_steps
            )
            * args.epochs
        )
        # if i == 0:
        #     print(model)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            # Prepare optimizer

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            weight_decay=0.01,
            correct_bias=False,
        )
        scheduler = WarmupLinearSchedule(
            optimizer,
            num_training_steps=num_train_optimization_steps,
            num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        )
        print("Training Model")

        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()

        model = torch.load(args.best_model_dir + "model.bin")
        eval_result = evaluate_split(model, processor, tokenizer, args, split="train")
        train_acc_list.append(eval_result[0]["train_accuracy"])
        train_f1_list.append(eval_result[0]["train_macro_f1"])

        eval_result = evaluate_split(model, processor, tokenizer, args, split="dev")
        dev_acc_list.append(eval_result[0]["dev_accuracy"])
        dev_f1_list.append(eval_result[0]["dev_macro_f1"])

        eval_result = evaluate_split(model, processor, tokenizer, args, split="test")
        test_acc_list.append(eval_result[0]["test_accuracy"])
        test_f1_list.append(eval_result[0]["test_macro_f1"])
        if (
            eval_result[0]["test_accuracy"] < 0.60
        ):  # keep the models with excellent performance
            shutil.rmtree(args.best_model_dir)
        else:
            print(
                "find eval_result[0][test_accuracy] >= 0.60",
                eval_result[0]["test_accuracy"],
                args.best_model_dir,
            )
            wandb.config.update(args, allow_val_change=True)
    from statistics import mean

    result = {}
    result["avg_train_acc"] = mean(train_acc_list)  # average of the ten runs
    result["avg_train_f1"] = mean(train_f1_list)  # average of the ten runs
    result["avg_dev_acc"] = mean(dev_acc_list)  # average of the ten runs
    result["avg_dev_f1"] = mean(dev_f1_list)  # average of the ten runs
    result["avg_test_acc"] = mean(test_acc_list)  # average of the ten runs
    result["test_acc_std"] = stdev(test_acc_list)  # average of the ten runs
    result["avg_test_f1"] = mean(test_f1_list)  # average of the ten runs
    result["test_f1_std"] = stdev(test_f1_list)  # average of the ten runs
    wandb.config.update(result)

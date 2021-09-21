import os
import random
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
from os import listdir
from statistics import mean, stdev

import numpy as np
import torch
from transformers import (
    AdamW,
    AdapterConfig,
    AdapterFusionConfig,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
)
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

import wandb
from utils.bert_evaluator import BertEvaluator
from utils.bert_trainer import BertTrainer
from utils.common_utils import print_args_as_table
from utils.medqa_processor import MedQaProcessor

wandb.init(project="MEDQA")


def get_args():
    parser = ArgumentParser(description="Evaluate model on MedQa dataset.")
    parser.add_argument(
        "--train_mode",
        default="fusion",
        type=str,
        required=True,
        help="three modes: fusion, adapter, base",
    )
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--base_model", default=None, type=str, required=True)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--cuda", action="store_true", help="to use gpu")
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat_runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain_epoch", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="t=1: softmax fusion, 0<t<1: gumbel softmax fusion, t<0: MOE",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1,
        help="training examples ratio to be kept.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--groups", type=str, default=None, help="groups to be chosen")

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--train_file", default="train.tsv")
    parser.add_argument("--dev_file", default="dev.tsv")
    parser.add_argument("--test_file", default="test.tsv")

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )

    args = parser.parse_args()
    return args


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


def prepare_opt_sch(model, args):
    """Prepare optimizer and scheduler.

    Args:
        model ([type]): [description]
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_examples = processor.get_train_examples()
    num_train_optimization_steps = (
        int(len(train_examples) / args.batch_size / args.gradient_accumulation_steps)
        * args.epochs
    )
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
    return optimizer, scheduler


def load_fusion_adapter_model(args):
    """Load fusion adapter model.

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    adapter_names_dict = search_adapters(args)
    base_model = AutoModelForMultipleChoice.from_pretrained(
        args.base_model, from_tf=get_tf_flag(args)
    )
    fusion_adapter_rename = []
    for model_path, adapter_names in adapter_names_dict.items():
        for adapter_name in adapter_names:
            adapter_dir = os.path.join(model_path, adapter_name)
            new_adapter_name = model_path[-14:][:-8] + "_" + adapter_name
            base_model.load_adapter(adapter_dir, load_as=new_adapter_name)
            print(f"Load adapter:{new_adapter_name}")
            fusion_adapter_rename.append(new_adapter_name)
    fusion_config = AdapterFusionConfig.load("dynamic", temperature=args.temperature)
    base_model.add_fusion(fusion_adapter_rename, fusion_config)
    base_model.set_active_adapters(fusion_adapter_rename)
    config = AutoConfig.from_pretrained(
        os.path.join(adapter_dir, "adapter_config.json")
    )
    # base_model.train_fusion([adapter_names])
    return config, base_model


def load_adapter_model(args):
    model_path = os.path.join(args.model_dir, args.model)
    base_model = AutoModelForMultipleChoice.from_pretrained(
        args.base_model, from_tf=get_tf_flag(args)
    )
    adapter_config = AdapterConfig.load(os.path.join(model_path, "adapter_config.json"))
    config = AutoConfig.from_pretrained(os.path.join(model_path, "adapter_config.json"))
    base_model.load_adapter(model_path, config=adapter_config)
    print(f"Load adapter:{config.name}")
    base_model.set_active_adapters([config.name])
    return config, base_model


if __name__ == "__main__":
    args = get_args()
    print(args)
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    model_str = args.model
    if "/" in model_str:
        model_str = model_str.split("/")[1]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Device:", str(device).upper())
    print("Number of GPUs:", n_gpu)
    print("AMP:", args.amp)
    if args.groups:
        args.groups = [int(i) for i in args.groups.split(",")]
        print("Groups:", args.groups)
    train_acc_list = []
    dev_acc_list = []
    test_acc_list = []
    seed_list = []
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.best_model_dir = f"./temp/model_{timestamp_str}/"
    # Record config on wandb
    wandb.config.update(args)
    print_args_as_table(args)

    processor = MedQaProcessor(args.data_dir)
    if args.tokenizer is None:
        args.tokenizer = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for i in range(args.repeat_runs):
        print(f"Start the {i}th training.")
        # Set random seed for reproducibility
        seed = int(time.time())
        print(f"Generate random seed {seed}.")
        seed_list.append(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        args.best_model_dir = f"./temp/model_{seed}/"
        os.makedirs(args.best_model_dir, exist_ok=True)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
        if args.train_mode == "fusion":
            # args.base_model will be a folder of pre-trained models over partitions
            config, model = load_fusion_adapter_model(args)
        elif args.train_mode == "adapter":
            # args.base_model will be a folder of a pre-trained model
            config, model = load_adapter_model(args)
        elif args.train_mode == "base":
            # use base bert model
            config = AutoConfig.from_pretrained(args.model)
            model = AutoModelForMultipleChoice.from_pretrained(
                args.model, from_tf=get_tf_flag(args)
            )

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        optimizer, scheduler = prepare_opt_sch(model, args)

        print("Training Model")
        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()
        print("Evaluating Model")
        model = torch.load(args.best_model_dir + "model.bin")
        train_result = evaluate_split(model, processor, tokenizer, args, split="train")
        train_result[0]["run_num"] = i
        wandb.log(train_result[0])  # Record Dev Result
        train_acc_list.append(train_result[0]["train_accuracy"])
        dev_result = evaluate_split(model, processor, tokenizer, args, split="dev")
        dev_result[0]["run_num"] = i
        wandb.log(dev_result[0])  # Record Dev Result
        dev_acc_list.append(dev_result[0]["dev_accuracy"])
        test_result = evaluate_split(model, processor, tokenizer, args, split="test")
        test_result[0]["run_num"] = i
        wandb.log(test_result[0])  # Record Testing Result
        test_acc_list.append(test_result[0]["test_accuracy"])
        if (
            test_result[0]["test_accuracy"] < 0.86
        ):  # keep the models with excellent performance
            shutil.rmtree(args.best_model_dir)
        else:
            print(f"Saving model to {args.best_model_dir}.")
            print(f"test_accuracy of {test_result[0]['test_accuracy']}.")

    result = {}
    result["seed_list"] = seed_list
    result["train_acc_mean"] = mean(train_acc_list)  # average of the ten runs
    result["train_acc_std"] = stdev(train_acc_list)  # average of the ten runs
    result["dev_acc_mean"] = mean(dev_acc_list)  # average of the ten runs
    result["dev_acc_std"] = stdev(dev_acc_list)  # average of the ten runs
    result["test_acc_mean"] = mean(test_acc_list)  # average of the ten runs
    result["test_acc_std"] = stdev(test_acc_list)  # average of the ten runs
    wandb.config.update(result)

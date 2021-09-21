from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(
        description="PyTorch deep learning models for document classification"
    )
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--base_model", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument("--tokenizer", default=None, type=str, required=False)
    parser.add_argument("--pretrain_epoch", default=None, type=str, required=False)
    parser.add_argument("--cuda", action="store_true", dest="cuda")
    parser.add_argument("--amp", action="store_true", help="use auto mixed precision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat_runs", type=int, default=10)

    parser.add_argument("--reduction_factor", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--add_sapbert", action="store_true")
    parser.add_argument("--add_rel_pred", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--groups", type=str, default=None)

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1,
        help="If <1, it will reduce the number of training examples",
    )
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--train_file", default="train.tsv")
    parser.add_argument("--dev_file", default="dev.tsv")
    parser.add_argument("--test_file", default="test.tsv")

    parser.add_argument(
        "--max_seq_length",
        default=128,
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

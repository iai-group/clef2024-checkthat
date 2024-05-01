"""Will run script to run training and testing. (test yet to be implemented)

Argument parser is used to specify the model name and dataset name.
"""
import argparse
from datasets import load_dataset
from training_scripts.training import run_training
from transformers import AutoTokenizer


def main(args):
    """Run training."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(args.dataset)
    label_map = {"No": 0, "Yes": 1}  # Label map for the dataset

    seeds = [42, 81, 1024, 6, 10]  # Seeds for reproducibility
    if args.train:
        for seed in seeds:
            run_training(seed, dataset, args.model_name, tokenizer, label_map)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run training and testing.")

    parser.add_argument(
        "--train", action="store_true", help="Whether to run training"
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to run testing"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="FacebookAI/roberta-large",  # For English language
        help="Name of the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iai-group/clef2024_checkthat_task1_en",  # For English language
        help="Name of the dataset",
    )

    args = parser.parse_args()
    main(args)

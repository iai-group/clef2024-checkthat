"""Will run script to run training and testing. (unlabeled tests yet to ble implemented)

Argument parser is used to specify the model name and dataset name.
"""
import argparse
from datasets import load_dataset
from training_scripts.training import run_training
from transformers import AutoTokenizer
from tokenization.tokenizer import TextDataset


def main(args):
    """Run training."""
    label_map = {"Yes": 1, "No": 0}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(args.dataset)

    dataset_language = args.dataset.split("_")[-2:]

    train_dataset = TextDataset(dataset["train"], tokenizer, label_map)
    eval_dataset = TextDataset(dataset["validation"], tokenizer, label_map)
    test_dataset = TextDataset(dataset["test"], tokenizer, label_map)

    run_training(train_dataset, eval_dataset, args.model_name, label_map, dataset_language, test_dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run training and testing.")

    # parser.add_argument(
    #     "--test", action="store_true", help="Whether to run testing"
    # )
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
        help="Name of the dataset from the iai-group/clef2024_checkthat_task1_* datasets",
    )
    
    args = parser.parse_args()
    main(args)

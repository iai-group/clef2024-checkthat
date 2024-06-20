"""Will run script to run training and testing. (unlabeled tests yet to ble
implemented)

Argument parser is used to specify the model name and dataset name.
"""
import argparse

from datasets import load_dataset
from tokenization.normalize_DatasetDict_featues import rename_features
from tokenization.tokenizer import TextDataset
from training_scripts.training import run_training
from transformers import AutoTokenizer


def main():
    # """Run training."""
    label_map = {"Yes": 1, "No": 0}

    """Load dataset and tokenizer."""
    model_name = "roberta-large"  # Replace this with the path to your tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("Model not found")

    dataset_name = "iai-group/clef2024_checkthat_task1_en"
    dataset = load_dataset(dataset_name)
    dataset_language = dataset_name[-2:]
    if "tweet_text" in dataset["train"].column_names:
        dataset = rename_features(dataset)

    train_dataset = TextDataset(dataset["train"], tokenizer, label_map)
    eval_dataset = TextDataset(dataset["validation"], tokenizer, label_map)
    test_dataset = TextDataset(dataset["test"], tokenizer, label_map)

    run_training(
        train_dataset,
        eval_dataset,
        model_name,
        label_map,
        dataset_language,
        test_dataset,
    )


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Run training and testing.")

    # # parser.add_argument(
    # #     "--test", action="store_true", help="Whether to run testing"
    # # )
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default="FacebookAI/roberta-large",  # For English language
    #     help="Name of the model",
    # )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="iai-group/clef2024_checkthat_task1_en",  # For English language
    #     help="Name of the dataset from the iai-group/clef2024_checkthat_task1_* datasets",
    # )

    # args = parser.parse_args()
    # main(args)
    main()

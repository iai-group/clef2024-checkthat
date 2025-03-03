"""Module for loading training configuration from a yaml file."""

import yaml
from transformers import TrainingArguments


def load_config(file_path) -> dict:
    """Load configuration from a yaml file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_training_arguments(model_name, seed, dataset_name) -> TrainingArguments:
    """Unpack training arguments from the config file and return as a
    TrainingArguments object, with dynamically adjusted output directory based
    on model name, seed, and dataset."""
    config = load_config("checkthat/task1/training_config.yaml")
    training_args = config["training_arguments"]

    # Extract a short language identifier from the dataset name
    language_code = dataset_name.split("_")[
        -1
    ]  # Assuming the dataset name ends with a language code

    # Modify the output_dir dynamically
    model_name_safe = model_name.replace("/", "_")
    training_args[
        "output_dir"
    ] = f"./results/{model_name_safe}_seed_{seed}_{language_code}"

    return TrainingArguments(**training_args)


def get_language(dataset_name) -> str:
    """Extract the language code from the dataset name."""
    dataset_language = dataset_name.split("_")[
        -1
    ]  # Assuming the dataset name ends with a language code
    return dataset_language

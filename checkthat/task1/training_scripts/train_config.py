"""Module to load training arguments from a yaml file."""

import yaml
from transformers import TrainingArguments


def load_config(file_path):
    """Load configuration from a yaml file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_training_arguments():
    """Unpack training arguments from the config file and return as a
    TrainingArguments object."""
    config = load_config("training_config.yaml")
    training_args = config["training_arguments"]
    return TrainingArguments(**training_args)

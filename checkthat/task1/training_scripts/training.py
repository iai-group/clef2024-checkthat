"""Training script for the model.

This script trains the model for a single seed.
"""
import wandb
from transformers import Trainer, EarlyStoppingCallback
from checkthat.task1.tokenization.tokenizer import TextDataset
from checkthat.task1.models.custom_model import CustomModel
from checkthat.task1.metrics.compute_metrics import compute_metrics
from checkthat.task1.training_scripts.train_config import get_training_arguments
import random
import numpy as np
import torch


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(seed, dataset, model_name, tokenizer, label_map):
    """Start training the model for a single seed.

    Args:
        seed: seed for reproducibility
        dataset: dataset dictionary containing train and validation splits
        model_name: huggingface model name
        tokenizer: huggerface tokenizer/same as model name
        label_map: dictionary mapping labels to integers
    """
    # Initialize wandb run
    set_seed(seed)
    run_name = f"{model_name}_{seed}"
    wandb.init(
        project="Clef2024",
        entity="aarnes",
        name=run_name,
        config={"seed": seed},
    )

    # Prepare datasets
    train_dataset = TextDataset(dataset["train"], tokenizer, label_map)
    eval_dataset = TextDataset(dataset["validation"], tokenizer, label_map)
    test_dataset = TextDataset(dataset["test"], tokenizer, label_map)

    training_arguments = get_training_arguments()
    training_arguments.run_name = (
        run_name  # Optional, sync the name with Trainer's internal wandb run
    )

    # Creating a Trainer instance with training arguments and datasets
    trainer = Trainer(
        model=CustomModel(model_name, num_labels=len(label_map)),
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],  # Early stopping callback
    )

    # Train the model
    trainer.train()

    # Finish the wandb run after each seed
    wandb.finish()

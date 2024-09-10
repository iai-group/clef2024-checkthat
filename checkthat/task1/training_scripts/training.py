"""Training script for the model.

This script trains the model for a single seed.
"""

import numpy as np
import torch
import torch.cuda
import wandb
import yaml

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


def compute_metrics(p: EvalPrediction) -> dict:
    """Compute metrics for evaluation."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", pos_label=1
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def load_config(file_path) -> dict:
    """Load configuration from a yaml file."""
    with open(file_path, "r") as file:
        full_config = yaml.safe_load(file)
    if full_config is None or "parameters" not in full_config:
        raise ValueError(
            "Configuration file is empty or incorrectly formatted."
        )
    # Simplify the config structure for easier usage:
    simplified_config = {
        key: value["values"][0]
        for key, value in full_config["parameters"].items()
    }
    return simplified_config


def run_training(
    train_dataset,
    eval_dataset,
    model_name,
    label_map,
    dataset_language,
    test_dataset=None,
) -> tuple:

    config = load_config("checkthat/task1/training_config.yaml")

    """Run training sweep.

    Evaluate on validation set and test set.
    """
    print("Starting to train..")
    run_name = wandb.init(
        project="wandb_yaml", entity="your_user", reinit=True
    ).name

    # Load model and tokenizer from Hugging Face
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_map)
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define training arguments
    training_arguments = TrainingArguments(
        output_dir=f"./results_{dataset_language}",
        evaluation_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=50,
        logging_dir="./logs",
        logging_steps=100,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=run_name,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=hf_model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer at the end of training
    model_path = (
        f"{training_arguments.output_dir}/{run_name}_model_{dataset_language}"
    )
    tokenizer_path = f"{training_arguments.output_dir}/{run_name}_tokenizer_{dataset_language}"

    hf_model.save_pretrained(model_path)
    hf_tokenizer.save_pretrained(tokenizer_path)

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Evaluate the model on the test dataset
    test_output = trainer.predict(test_dataset)
    test_results = {f"test_{k}": v for k, v in test_output.metrics.items()}

    # Log evaluation and test results to W&B
    wandb.log({"eval_results": eval_results})
    wandb.log({"test_results": test_results})
    # Ensure the W&B run is finished
    wandb.finish()

    # Return paths for model and tokenizer for user reference
    return model_path, tokenizer_path

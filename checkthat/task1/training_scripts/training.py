"""Training script for the model.

This script trains the model for a single seed.
"""
import wandb
from transformers import Trainer, EarlyStoppingCallback
from tokenization.tokenizer import TextDataset
from models.custom_model import CustomModel
from metrics.compute_metrics import compute_metrics
from training_scripts.train_config import get_training_arguments
from training_scripts.train_config import get_language
import numpy as np
import torch
import torch.cuda
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments



def run_training(train_dataset, eval_dataset, model_name, label_map, dataset_language, test_dataset=None,):
    """Run training sweep. Evaluate on validation set and test set."""
    
    run_name = wandb.init(
        project="sweep_test",
        entity="aarnes",
        reinit=True
    ).name

    # Load model and tokenizer from Hugging Face
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map))
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Define training arguments
    training_arguments = TrainingArguments(
        output_dir="./results",  # Directory to save model and tokenizer
        evaluation_strategy="epoch",
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.batch_size,
        num_train_epochs=wandb.config.epochs,
        logging_dir='./logs',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        save_strategy="epoch",  # Save model at the end of each epoch
        save_total_limit=1,  # Optional: limits the total amount of checkpoints, deleting older
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()
    # Evaluate the model on the test dataset
    test_output = trainer.predict(test_dataset)
    test_results = {f"test_{k}": v for k, v in test_output.metrics.items()}

    

    # Evaluate the model
    eval_results = trainer.evaluate()
    
    # Log evaluation and test results to W&B
    wandb.log({"eval_results": eval_results})
    wandb.log({"test_results": test_results})

    # Save model and tokenizer at the end of training
    model_path = f"{training_arguments.output_dir}/{run_name}_model_{dataset_language}"
    tokenizer_path = f"{training_arguments.output_dir}/{run_name}_tokenizer_{dataset_language}"

    hf_model.save_pretrained(model_path)
    hf_tokenizer.save_pretrained(tokenizer_path)

    # Ensure the W&B run is finished
    wandb.finish()

    # Return paths for model and tokenizer for user reference
    return model_path, tokenizer_path
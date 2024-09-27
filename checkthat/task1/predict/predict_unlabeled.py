import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import argparse


def main(args) -> None:
    """Run inference on unlabeled test data."""
    label_map = {0: "No", 1: "Yes"}

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_map)
    )

    # Assuming you're using a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load your data from a TSV file
    input_data = pd.read_csv(args.input_data, sep="\t")
    dataset = Dataset.from_pandas(input_data)

    # Prepare dataset for processing
    def tokenize_function(examples):
        return tokenizer(
            examples["Text"], padding="max_length", truncation=True
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    test_dataset = tokenized_datasets.remove_columns(["Text"])
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "Sentence_id"]
    )

    # DataLoader setup
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset, batch_size=16, collate_fn=data_collator
    )

    # Run predictions
    run_id = args.model_name.split("/")[
        -1
    ]  # Use the model name as the run identifier
    results = []

    for batch in test_loader:
        batch_inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }  # Filtering out non-model inputs
        with torch.no_grad():
            outputs = model(**batch_inputs)
            pred_labels = torch.argmax(outputs.logits, dim=-1)

            # Fetch sentence IDs for the current batch
            ids = batch["Sentence_id"].cpu().numpy()

            results.extend(
                zip(ids, pred_labels.cpu().numpy(), [run_id] * len(ids))
            )

    # Write predictions to a TSV file
    output_file = f"unlabeled_test_results_{run_id}.tsv"
    with open(output_file, "w") as file:
        file.write("id\tclass_label\trun_id\n")
        for result in results:
            file.write(f"{result[0]}\t{label_map[result[1]]}\t{result[2]}\n")

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on unlabeled test data."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large",
        help="Name or path of the model to use",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="unlabeled_test_data.tsv",
        help="Path to the input TSV file",
    )
    args = parser.parse_args()
    main(args)

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def main():
    label_map = {0: "No", 1: "Yes"}
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("/home/stud/u2929246/bhome/checkthat_sweep/clef2024-checkthat/checkthat/task1/results_en/astral-sweep-10_tokenizer_en")
    model_path = "/home/stud/u2929246/bhome/checkthat_sweep/clef2024-checkthat/checkthat/task1/results_en/astral-sweep-10_model_en"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label_map))

    # Assuming you're using a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load your data from a TSV file
    input_data = pd.read_csv("/home/stud/u2929246/bhome/checkthat_sweep/clef2024-checkthat/checkthat/task1/unlabeled_test.tsv", sep='\t')
    dataset = Dataset.from_pandas(input_data)

    # Prepare dataset for processing
    def tokenize_function(examples):
        return tokenizer(examples['Text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    test_dataset = tokenized_datasets.remove_columns(["Text"])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Sentence_id'])

    # DataLoader setup
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

    # Run predictions
    run_id = "roberta-large"  # Replace this with your actual run identifier
    results = []

    for batch in test_loader:
        batch_inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}  # Filtering out non-model inputs
        with torch.no_grad():
            outputs = model(**batch_inputs)
            pred_labels = torch.argmax(outputs.logits, dim=-1)

            # Fetch sentence IDs for the current batch
            ids = batch['Sentence_id'].cpu().numpy()  # Extracting Sentence IDs from the batch

            results.extend(zip(ids, pred_labels.cpu().numpy(), [run_id]*len(ids)))

    # Write predictions to a TSV file
    with open('results.tsv', 'w') as file:
        file.write("id\tclass_label\trun_id\n")
        for result in results:
            file.write(f"{result[0]}\t{label_map[result[1]]}\t{result[2]}\n")

if __name__ == "__main__":
    main()

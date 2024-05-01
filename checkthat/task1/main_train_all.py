"""Main training script for training on all languages."""
from datasets import load_dataset
from tokenization.normalize_DatasetDict_featues import rename_features
from training_scripts.training import run_training
from transformers import AutoTokenizer


def main():
    en, ar, es, nl = (
        "iai-group/clef2024_checkthat_task1_en",
        "iai-group/clef2024_checkthat_task1_ar",
        "iai-group/clef2024_checkthat_task1_es",
        "iai-group/clef2024_checkthat_task1_nl",
    )

    dataset_list = [en, ar, es, nl]
    label_map = {"No": 0, "Yes": 1}  # Label map for the dataset

    model_name_en = "FacebookAI/roberta-large"
    multilingual_model = "FacebookAI/xlm-roberta-large"

    seeds = [42, 81, 1024, 6, 10]  # Seeds for reproducibility

    tokenizer = AutoTokenizer.from_pretrained(model_name_en)

    for dataset in dataset_list:
        for seed in seeds:
            dataset = load_dataset(dataset)
            # Normalize dataset features if not already normalized (intended for twitter dataset)
            if "tweet_text" in dataset["train"].column_names:
                dataset = rename_features(dataset)
                tokenizer = AutoTokenizer.from_pretrained(multilingual_model)
                run_training(seed, dataset, multilingual_model, tokenizer, label_map)
            else:
                run_training(seed, dataset, model_name_en, tokenizer, label_map)


if __name__ == "__main__":
    import torch

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    main()

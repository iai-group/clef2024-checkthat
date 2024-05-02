"""Main training script for training on all languages."""
from datasets import load_dataset
from tokenization.normalize_DatasetDict_featues import rename_features
from training_scripts.training import run_training
from transformers import AutoTokenizer
from training_scripts.train_config import get_training_arguments
from test_scripts.load_from_checkpoints import load_model_from_dir
from test_scripts.run_tests import run_testing

def main():
    dataset_list = [
        "iai-group/clef2024_checkthat_task1_en",
        "iai-group/clef2024_checkthat_task1_ar",
        "iai-group/clef2024_checkthat_task1_es",
        "iai-group/clef2024_checkthat_task1_nl",
    ]
    label_map = {"Yes": 1, "No": 0}

    model_name_en = "FacebookAI/roberta-large"
    multilingual_model = "FacebookAI/xlm-roberta-large"
    seeds = [42, 81, 1024, 6, 10]
    tokenizer = AutoTokenizer.from_pretrained(model_name_en)

    """Training model on trainset for each seed and each language"""
    for seed in seeds:
        for dataset_name in dataset_list:
            dataset = load_dataset(dataset_name)
            if "tweet_text" in dataset["train"].column_names:
                dataset = rename_features(dataset)
                tokenizer = AutoTokenizer.from_pretrained(multilingual_model)
                training_args = get_training_arguments(multilingual_model, seed, dataset_name)
                run_training(seed, dataset, model_name_en, tokenizer, label_map, training_args)
            else:
                training_args = get_training_arguments(model_name_en, seed, dataset_name)
                run_training(seed, dataset, model_name_en, tokenizer, label_map, training_args)



    """Testing model on testset"""
    base_dir = "./results"
    models = load_model_from_dir(base_dir)
    for model_name, model in models.items():
        i += 1 # Incrementing i to get the model name for each model
        for dataset_name in dataset_list:
            dataset = load_dataset(dataset_name)
            if "tweet_text" in dataset["test"].column_names:
                dataset = rename_features(dataset)
                tokenizer = AutoTokenizer.from_pretrained(multilingual_model)
                run_testing(model, dataset, tokenizer, label_map)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                run_testing(model, dataset, tokenizer, label_map, model.keys()[i]) # model.keys()[i] to get the model name

if __name__ == "__main__":
    import torch

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    main()

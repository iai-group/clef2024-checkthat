from load_from_checkpoints import load_model_from_dir


base_dir = "./results"
models = load_model_from_dir(base_dir)


dataset_list = [
    "iai-group/clef2024_checkthat_task1_en",
    "iai-group/clef2024_checkthat_task1_ar",
    ]



dataset = load_dataset(dataset_name)

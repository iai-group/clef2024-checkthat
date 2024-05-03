import torch
from models.custom_model import CustomModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

path = "checkthat/task1/test_model_xlm_roberta/model.safetensors"

print("test")
print(path)

model = CustomModel("xlm-roberta-base", 2, "cpu")
model.load_state_dict(torch.load(path, map_location="cpu"))


# config = AutoConfig.from_pretrained(path)
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# model = AutoModelForSequenceClassification.from_pretrained(path, config=config)


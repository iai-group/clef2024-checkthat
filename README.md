# CLEF 2024 CheckThat! task1 IAI participation.

This repo contains the code and data for the CLEF 2024 CheckThat! task1 IAI participation.

## Folders
* checkthat - python module for the claim detection.
* data
  * task1 - Contains data for task1 which is also uploaded to HuggingFace framework.
    * The dataset is currently gated/private, make sure you have run huggingface-cli login
    * Usage: ```
        from datasets import load_dataset
        # English data containing political debates.
        dataset_en = load_dataset("iai-group/clef2024_checkthat_task1_en")
        # Spanish data containing Tweets.
        dataset_es = load_dataset("iai-group/clef2024_checkthat_task1_es")
        # Dutch data containing Tweets.
        dataset_nl = load_dataset("iai-group/clef2024_checkthat_task1_nl")
        # Arabic data containing Tweets.
        dataset_ar = load_dataset("iai-group/clef2024_checkthat_task1_ar")
    ```
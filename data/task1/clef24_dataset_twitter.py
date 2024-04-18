"""Multilang Dataset loading script."""

from datasets import (
    DatasetInfo,
    BuilderConfig,
    Version,
    GeneratorBasedBuilder,
    DownloadManager,
)
from datasets import SplitGenerator, Split, Features, Value
from typing import Generator, Tuple, Union

import os

_DESCRIPTION = """
This dataset includes Arabic/Dutch/Spanish Twitter data for CLEF 2024 CheckThat! Lab task1. 
"""  # noqa E501

_CITATION = """\
@inproceedings{barron2024clef,
  title={The CLEF-2024 CheckThat! Lab: Check-Worthiness, Subjectivity, Persuasion, Roles, Authorities, and Adversarial Robustness},
  author={Barr{\'o}n-Cede{\~n}o, Alberto and Alam, Firoj and Chakraborty, Tanmoy and Elsayed, Tamer and Nakov, Preslav and Przyby{\l}a, Piotr and Stru{\ss}, Julia Maria and Haouari, Fatima and Hasanain, Maram and Ruggeri, Federico and others},
  booktitle={European Conference on Information Retrieval},
  pages={449--458},
  year={2024},
  organization={Springer}
}
"""  # noqa E501

_LICENSE = "Your dataset's license here."


class CLEF24TwitterData(GeneratorBasedBuilder):
    """A multilingual Twitter dataset for claim detection."""

    BUILDER_CONFIGS = [
        BuilderConfig(
            name="clef24_tweet_data",
            version=Version("1.0.0"),
            description="Multilingual dataset for text classification.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "clef24_tweet_data"  # Default configuration name.

    def _info(self):
        """Construct the DatasetInfo object."""
        return DatasetInfo(
            description=_DESCRIPTION,
            features=Features(
                {
                    "tweet_id": Value("string"),
                    "tweet_url": Value("string"),
                    "tweet_text": Value("string"),
                    "class_label": Value("string"),
                }
            ),
            supervised_keys=("tweet_text", "class_label"),
            homepage="https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task1",  # noqa E501
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(
        self, dl_manager: DownloadManager
    ) -> list[SplitGenerator]:
        """Returns SplitGenerators."""
        # Assumes your dataset is located in "."
        data_dir = os.path.abspath(".")
        splits = {
            "train": Split.TRAIN,
            "dev": Split.VALIDATION,
            "test": Split.TEST,
        }

        return [
            SplitGenerator(
                name=splits[split],
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{split}.tsv"),
                    "split": splits[split],
                },
            )
            for split in splits.keys()
        ]

    def _generate_examples(
        self, filepath: Union[str, os.PathLike], split: str
    ) -> Generator[Tuple[str, dict], None, None]:
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                if id_ == 0:  # Optionally skip header
                    continue
                cols = row.strip().split("\t")
                yield f"{split}_{id_}", {
                    "tweet_id": cols[0],
                    "tweet_url": cols[1],
                    "tweet_text": cols[2],
                    "class_label": cols[3],
                }

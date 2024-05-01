"""For the datasets that do not follow the english dataset format, we need to
rename the features to match the english dataset format."""


def rename_features(data) -> dict:
    """Hacky function intended to use for twitter data to it uses same features
    as other english dataset."""
    # Iterate over each split (train, validation, test)
    feature_name_mapping = {
        "tweet_text": "Text",
    }
    for split_name in data.keys():
        # Get the dataset for the current split
        split_dataset = data[split_name]

        # Rename each feature in the dataset using the mapping
        for old_name, new_name in feature_name_mapping.items():
            split_dataset = split_dataset.rename_column(old_name, new_name)

        # Update the dataset in the DatasetDict
        data[split_name] = split_dataset
    return data

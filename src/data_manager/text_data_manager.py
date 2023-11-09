import re
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from src.case import Case


#     return tokenized_output["input_ids"].reshape(-1, seq_length)
def filter_and_tokenize_sentences(
    text: str, max_token_length: int, tokenizer
) -> List[str]:
    """
    Filters sentences in a text based on the maximum token length.
    Return the tokenize sentences.

    Parameters:
    - text (str): The text to be filtered.
    - max_token_length (int): The maximum allowed token length.
    - tokenizer: The tokenizer used to convert sentences to tokens.

    Returns:
    - List[str]: A list of filtered sentences.
    """
    sentences = sent_tokenize(text)
    # Tokenize in batches
    tokenized_sentences = tokenizer(
        sentences,
        add_special_tokens=False,
        return_attention_mask=False,
        padding=False,
        truncation=False,
    )

    # Filter out tokenized sentences that are too long
    filtered_tokenized_ids = [
        torch.tensor(token_ids)
        for token_ids in tokenized_sentences["input_ids"]
        if len(token_ids) <= max_token_length
    ]

    return filtered_tokenized_ids


def get_id_from_data(
    data: List[str], seq_length: int, tokenizer
) -> torch.Tensor:
    unpadded_tokenized_ids = []
    for text in tqdm(data, desc="Preparing dataset"):
        unpadded_tokenized_ids.extend(
            filter_and_tokenize_sentences(text, seq_length, tokenizer)
        )
    tokenizer.pad_token = tokenizer.eos_token
    padded_tokenized_sentences = pad_sequence(
        unpadded_tokenized_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    return padded_tokenized_sentences


def get_wiki_data(
    seq_length: int, tokenizer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes training and testing datasets by cleaning, tokenizing, and filtering them.

    Parameters:
    - seq_length (int): The sequence length to be used for tokenizing.
    - tokenizer: The tokenizer used for tokenizing the text.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: Tensors of tokenized training and testing text data.
    """

    def clean_and_filter(data: List[str]) -> List[str]:
        cleaned_data = [
            re.sub(
                r"\s+([.,:;?!])",
                r"\1",
                re.sub(r"@-@", "-", re.sub(r" @,@ ", "", text)),
            )
            for text in data
            if text.strip() != ""
        ]
        return [
            string
            for string in cleaned_data
            if not string.strip().startswith("=")
            and not string.strip().endswith("=\n")
        ]

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # Put this in a function which return train and test data
    train_data = dataset["train"]["text"]
    test_data = dataset["test"]["text"]
    # Process the train data
    train_cleaned = clean_and_filter(train_data)
    train_tensor = get_id_from_data(train_cleaned, seq_length, tokenizer)

    # Process the test data
    test_cleaned = clean_and_filter(test_data)
    test_tensor = get_id_from_data(test_cleaned, seq_length, tokenizer)

    return train_tensor, test_tensor


def get_lm1b_data(seq_length, tokenizer, short=False):
    if short:
        dataset_stream_train = load_dataset(
            "lm1b", split="train", streaming=True
        )
        data_train = list(dataset_stream_train.take(int(2e5)))

        dataset_stream_test = load_dataset(
            "lm1b", split="test", streaming=True
        )
        data_test = list(dataset_stream_test.take(int(2e4)))
    else:
        data_train = load_dataset("lm1b", split="train")
        data_test = load_dataset("lm1b", split="test")

    train_sentences = [item["text"] for item in data_train]
    test_sentences = [item["text"] for item in data_test]

    train_data = get_id_from_data(train_sentences, seq_length, tokenizer)
    test_data = get_id_from_data(test_sentences, seq_length, tokenizer)

    return train_data, test_data


def prepare_text_dataset(
    name: str, seq_length: int, tokenizer_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares a text dataset for training and testing.

    Parameters:
    - name (str): The name of the dataset.
    - seq_length (int): The sequence length to be used for tokenizing.
    - tokenizer_name (str): The name of the tokenizer.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: Tokenized training and testing data.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if name == Case.wiki:
        train_data, test_data = get_wiki_data(seq_length, tokenizer)
    elif name == Case.lm1b_short:
        train_data, test_data = get_lm1b_data(
            seq_length, tokenizer, short=True
        )
    elif name == Case.lm1b:
        train_data, test_data = get_lm1b_data(seq_length, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_data, test_data


def decode_tokens(x, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if x.dim() == 2:
        decoded_x = []
        for xi in x:
            decoded_text = tokenizer.decode(xi, skip_special_tokens=True)
            decoded_x.append(decoded_text)
        return decoded_x
    else:
        return tokenizer.decode(x, skip_special_tokens=True)


def decode_list_tokens(list, tokenizer_name):
    list_decoded = []
    for xi in list:
        list_decoded.append(decode_tokens(xi, tokenizer_name))
    return list_decoded


def get_nb_tokens(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.vocab_size

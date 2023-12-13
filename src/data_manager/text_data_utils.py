import re
import os
import torch
from tqdm import tqdm
from typing import List, Tuple, Union
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence

from src.case import Case


class TextDataUtils:
    is_nltk_downloaded = False

    @staticmethod
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
        from nltk.tokenize import sent_tokenize

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

    @staticmethod
    def get_id_from_data(
        data: List[str], seq_length: int, tokenizer: PreTrainedTokenizer
    ) -> torch.Tensor:
        """
        Processes a list of text data into padded tokenized IDs.

        If NLTK is not already downloaded, it will be downloaded during the first call of this method.

        Parameters:
            data (List[str]): A list of text strings to be tokenized.
            seq_length (int): The sequence length to be considered for tokenization.
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.

        Returns:
            torch.Tensor: A tensor of padded tokenized sentence IDs.

        Note:
            The tokenizer's pad_token is set to its eos_token.
        """
        if not TextDataUtils.is_nltk_downloaded:
            import nltk

            nltk.download("punkt")
            TextDataUtils.is_nltk_downloaded = True

        unpadded_tokenized_ids = []
        for text in tqdm(data, desc="Preparing dataset"):
            unpadded_tokenized_ids.extend(
                TextDataUtils.filter_and_tokenize_sentences(text, seq_length, tokenizer)
            )
        tokenizer.pad_token = tokenizer.eos_token
        padded_tokenized_sentences = pad_sequence(
            unpadded_tokenized_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        return padded_tokenized_sentences

    @staticmethod
    def get_wiki_data(seq_length: int, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
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
        train_tensor = TextDataUtils.get_id_from_data(
            train_cleaned, seq_length, tokenizer
        )

        # Process the test data
        test_cleaned = clean_and_filter(test_data)
        test_tensor = TextDataUtils.get_id_from_data(
            test_cleaned, seq_length, tokenizer
        )

        return train_tensor, test_tensor

    @staticmethod
    def get_lm1b_data(
        seq_length: int, tokenizer: AutoTokenizer, short: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and processes the LM1B dataset for training and testing.

        Parameters:
            seq_length (int): The sequence length for tokenization.
            tokenizer (AutoTokenizer): The tokenizer for processing the text data.
            short (bool, optional): Flag to load a smaller subset of the dataset. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing the tokenized training and testing data.
        """
        if short:
            dataset_stream_train = load_dataset("lm1b", split="train", streaming=True)
            data_train = list(dataset_stream_train.take(int(1e6)))

            dataset_stream_test = load_dataset("lm1b", split="test", streaming=True)
            data_test = list(dataset_stream_test.take(int(1e5)))
        else:
            data_train = load_dataset("lm1b", split="train")
            data_test = load_dataset("lm1b", split="test")

        train_sentences = [item["text"] for item in data_train]
        test_sentences = [item["text"] for item in data_test]

        train_data = TextDataUtils.get_id_from_data(
            train_sentences, seq_length, tokenizer
        )
        test_data = TextDataUtils.get_id_from_data(
            test_sentences, seq_length, tokenizer
        )

        return train_data, test_data

    @staticmethod
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
            train_data, test_data = TextDataUtils.get_wiki_data(seq_length, tokenizer)
        elif name == Case.lm1b_short:
            train_data, test_data = TextDataUtils.get_lm1b_data(
                seq_length, tokenizer, short=True
            )
        elif name == Case.lm1b:
            train_data, test_data = TextDataUtils.get_lm1b_data(seq_length, tokenizer)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        return train_data, test_data

    @staticmethod
    def decode_tokens(x: torch.Tensor, tokenizer_name: str) -> Union[str, List[str]]:
        """
        Decodes tokens from a tensor using a specified tokenizer.

        Parameters:
            x (torch.Tensor): A tensor of token IDs.
            tokenizer_name (str): The name of the tokenizer for decoding.

        Returns:
            Union[str, List[str]]: Decoded text or a list of decoded texts.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if x.dim() == 2:
            decoded_x = []
            for xi in x:
                decoded_text = tokenizer.decode(xi, skip_special_tokens=True)
                decoded_x.append(decoded_text)
            return decoded_x
        else:
            return tokenizer.decode(x, skip_special_tokens=True)

    @staticmethod
    def decode_list_tokens(
        tokens_list: List[torch.Tensor], tokenizer_name: str
    ) -> List[Union[str, List[str]]]:
        """
        Decodes a list of token tensors using a specified tokenizer.

        Parameters:
            tokens_list (List[torch.Tensor]): A list of tensors containing token IDs.
            tokenizer_name (str): The name of the tokenizer for decoding.

        Returns:
            List[Union[str, List[str]]]: A list of decoded texts or lists of decoded texts.
        """
        return [TextDataUtils.decode_tokens(xi, tokenizer_name) for xi in tokens_list]

    @staticmethod
    def get_nb_tokens(tokenizer_name: str) -> int:
        """
        Retrieves the number of tokens in the tokenizer's vocabulary.

        Parameters:
            tokenizer_name (str): The name of the tokenizer.

        Returns:
            int: The size of the tokenizer's vocabulary.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer.vocab_size

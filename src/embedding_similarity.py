#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AutoModel
from transformers import AutoTokenizer

cache_bert_embedding = {}


def get_bert_embedding(model_name: str) -> BertEmbedding:
    global cache_bert_embedding
    model_name = os.path.join("/model", model_name)
    if model_name not in cache_bert_embedding:
        cache_bert_embedding[model_name] = BertEmbedding(tokenizer_name=model_name)
    return cache_bert_embedding[model_name]


class BertEmbedding:
    """Get the embedding from Bert model using transformers library

    Parameters
    ----------
    tokenizer_name: str
        Model are available here : https://huggingface.co/transformers/pretrained_models.html
        & https://huggingface.co/models

    model_name: str default None
        Specify a different model from the tokenizer, if None it will load the tokenize related model
        Use this with your own pretrained model

    Attributes
    ----------
    tokenizer : tokenizer from transformers
    model: model from transformers

    """

    def __init__(self, tokenizer_name, model_name=None) -> None:
        if model_name is None:
            model_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _format_input(self, text, batch_size: int = 100):
        """ Tokenize text & encode to DataLoader

        text: list(str) or pd.Series(str) or str
            text to embed
        batch_size : int default 100, smaller batch to fit on RAM
        """
        # tokenize text
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Format to DataLoader to use batch_size (for small cpu/gpu memory)
        data = TensorDataset(encoded_input["input_ids"], encoded_input["attention_mask"])
        sampler = SequentialSampler(data)
        text_dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return text_dataloader

    def run_model(self, text: Iterable[str], batch_size: int = 100, mlm_model: bool = False) -> torch.Tensor:
        """Encode text and output sentence embedding

        Parameters
        ----------
        text: list(str) or pd.Series(str) or str
            text to embed
        batch_size : int default 100, smaller batch to fit on RAM
        mlm_model: boolean
            model is fine tune on task MLM (Masked Language Modeling) or not
            In this case the outputs of the model are different from a classic pretrained model

        Returns
        -------
        sentence_embeddings : torch.Tensor(x, 768)

        """
        text_dataloader = self._format_input(text, batch_size)

        self.model.eval()
        # Compute token embeddings
        sentence_embeddings = None

        for batch in text_dataloader:
            b_input_ids = batch[0]
            b_attention_mask = batch[1]

            with torch.no_grad():
                model_output = self.model(b_input_ids, b_attention_mask, output_hidden_states=mlm_model)

            if mlm_model:
                token_embeddings = model_output[1][-1]  # corresponding to last hidden state before MLM
            else:
                token_embeddings = None

            # Perform pooling on the model output
            embed = _mean_pooling(model_output=model_output,
                                  attention_mask=b_attention_mask,
                                  token_embeddings=token_embeddings)
            # Cat tensor batch
            try:
                sentence_embeddings = torch.cat((sentence_embeddings, embed))
            except TypeError:  # Initialise sentence_embeddings
                sentence_embeddings = embed

        return sentence_embeddings


def _mean_pooling(model_output, attention_mask, token_embeddings=None):
    """Mean Pooling - Take attention mask into account for correct averaging (ignore padding)
        ie: word (or token) embedding to sentence embedding
    """
    if token_embeddings is None:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

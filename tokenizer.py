# tokenizer.py

import os
from logging import getLogger
from typing import List, Dict, Sequence

import tiktoken
from tiktoken.load import load_tiktoken_bpe

logger = getLogger(__name__)

class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        self.n_words = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",  # end of turn
        ]
        self.special_tokens = {
            token: self.n_words + i for i, token in enumerate(special_tokens)
        }
        self.n_words += len(special_tokens)

        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self.model = tiktoken.Encoding(
            name=os.path.basename(model_path),
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = self.n_words - 1
        self.stop_tokens = {self.eos_id, self.special_tokens["<|eot_id|>"]}

        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self.model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(t)

    def encode_dialog(self, dialog: List[Dict[str, str]]) -> List[int]:
        tokens = [self.bos_id]
        for message in dialog:
            role = message["role"]
            content = message["content"]
            tokens.extend(self.encode_message(role, content))
        tokens.extend(self.encode_message("assistant", ""))
        return tokens

    def encode_message(self, role: str, content: str) -> List[int]:
        tokens = [
            self.special_tokens["<|start_header_id|>"],
            *self.encode(role, bos=False, eos=False),
            self.special_tokens["<|end_header_id|>"],
            *self.encode("\n\n", bos=False, eos=False),
            *self.encode(content.strip(), bos=False, eos=False),
            self.special_tokens["<|eot_id|>"],
        ]
        return tokens

Dialog = List[Dict[str, str]]

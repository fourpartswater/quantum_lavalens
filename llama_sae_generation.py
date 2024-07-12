# llama_sae_generation.py
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama_sae_model import LlamaSaeModel, ModelArgs
from tokenizer import Tokenizer, Dialog

def load_model(ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int, sae_layers: Optional[List[int]] = None):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = LlamaSaeModel(model_args, sae_layers)
    
    # Load Llama checkpoint
    checkpoint = torch.load(checkpoints[0], map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    # Load SAE checkpoints if specified
    if sae_layers:
        for layer in sae_layers:
            sae_ckpt = torch.load(f"path/to/sae/checkpoints/layer_{layer}.pth", map_location="cpu")
            model.saes[f"layer_{layer}"].load_state_dict(sae_ckpt)

    model.to(local_rank)

    return model, tokenizer

class LlamaSaeGenerator:
    def __init__(self, model: LlamaSaeModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> List[List[int]]:
        bsz = len(prompt_tokens)
        params = self.model.params

        total_len = min(params.max_seq_len, max_gen_len + max(len(t) for t in prompt_tokens))

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        for cur_pos in range(min(len(t) for t in prompt_tokens), total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            toks = toks[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
            out_tokens.append(toks)
        return out_tokens

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ) -> List[str]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return [self.tokenizer.decode(t) for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ) -> List[str]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [
            self.tokenizer.encode_dialog(dialog) for dialog in dialogs
        ]
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return [self.tokenizer.decode(t) for t in generation_tokens]

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

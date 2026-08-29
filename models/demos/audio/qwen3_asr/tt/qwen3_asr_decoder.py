# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn Qwen3-1.7B decoder for Qwen3-ASR, built on tt_transformers.

The text decoder is a standard Qwen3 (validated: extracted checkpoint reproduces
golden logits PCC=1.0). We reuse `tt_transformers.tt.model.Transformer` verbatim; the
only model-level change is the qwen3_vl "tokens is actually embeddings" trick in
`prepare_inputs_prefill`, so the prompt enters as pre-merged embeddings (audio embeds
spliced at the audio-token positions) instead of token ids — the qwen3_vl pattern,
minus vision MRoPE (Qwen3-ASR uses plain 1D RoPE).

Both prefill and the greedy decode loop are driven through the shared
`tt_transformers.tt.generator.Generator` rather than bespoke plumbing:
  - prefill: `Generator.prefill_forward_single_user_text` (single-user, non-paged),
    which calls our embeds-aware `prepare_inputs_prefill` + the shared `ttnn_prefill_forward`.
  - decode : `Generator.decode_forward(enable_trace=False, ...)` + host argmax greedy.

Trace is intentionally left OFF: a persistent decode trace destabilized the long-lived
server across mixed prefill lengths (see README "Known limitations"). Generator makes
trace opt-in per call, so we keep the shared decode path without the instability.

prefill (embeds) -> greedy decode loop (token ids) -> text.
"""
import os

import torch

import ttnn
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.common import get_block_size, num_blocks_in_seq
from models.tt_transformers.tt.model import Transformer


# Weight dtype for the text decoder. A serving stack that builds the decoder with
# bfloat8_b and a demo that builds it with bfloat16 decode the same clip
# differently ("コース" instead of "構成"), which looks like a model bug but is
# just a different quantisation. Default to bfloat8_b and make it explicit.
_DTYPES = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}


def decoder_weight_dtype():
    """ttnn dtype for the decoder weights, overridable via QWEN3ASR_DECODER_DTYPE."""
    name = os.environ.get("QWEN3ASR_DECODER_DTYPE", "bfloat8_b")
    if name not in _DTYPES:
        raise ValueError(f"QWEN3ASR_DECODER_DTYPE must be one of {sorted(_DTYPES)}, got {name!r}")
    return _DTYPES[name]


# Capture a decode trace. Without it every decode step pays full per-op host
# dispatch: measured 489 ms/token untraced on p150 versus 113 ms traced, which is
# what makes an untraced front-end look ~4x slower than a traced serving path for
# the same model. Override with QWEN3ASR_DECODE_TRACE=0 if a deployment hits the
# long-run trace instability noted in the README.
DECODE_TRACE = os.environ.get("QWEN3ASR_DECODE_TRACE", "1").strip().lower() in ("1", "true", "yes", "on")


# Stop ids from the checkpoint's generation_config ("eos_token_id": [151643,
# 151645]). Callers that stop on only one of them keep decoding past a valid stop
# and emit tokens the reference implementation never produces.
EOS_TOKEN_IDS = (151643, 151645)


def apply_repetition_penalty(logits, seen_ids, penalty):
    """vLLM repetition penalty, applied in place on a 1-D logit row.

    Positive logits are divided by ``penalty`` and negative ones multiplied, for
    every token id in ``seen_ids``.

    ``seen_ids`` must be the union of the PROMPT token ids and the ids generated
    so far. vLLM's ``apply_penalties`` builds ``prompt_mask | output_mask`` and
    penalises both (see ``_custom_ops.apply_repetition_penalties_torch``);
    penalising only the generated ids is HF transformers' behaviour and decodes
    differently.
    """
    if penalty is None or penalty == 1.0 or seen_ids is None or len(seen_ids) == 0:
        return logits
    ids = torch.tensor(sorted({int(i) for i in seen_ids}), dtype=torch.long)
    ids = ids[ids < logits.shape[-1]]
    if ids.numel() == 0:
        return logits
    selected = logits[ids]
    logits[ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


class Qwen3ASRDecoder(Transformer):
    @property
    def generator(self):
        """Lazily wrap this model in a stock `Generator` (single replica) used to drive
        prefill + decode. Constructed on first use so existing callers (which build the
        decoder exactly like any tt_transformers model) need no changes."""
        gen = getattr(self, "_generator", None)
        if gen is None:
            gen = self._generator = Generator([self], [self.args], self.mesh_device)
        return gen

    def prepare_inputs_prefill(self, tokens, **kwargs):
        """qwen3_vl-style embeds injection ("tokens is actually embeddings").

        `tokens` here is the pre-merged inputs_embeds (torch, (1, S, dim) or (S, dim)).
        We reuse the base `Transformer.prepare_inputs_prefill` for all the rot-mat /
        page-table preparation — driven by a throwaway id tensor — and only swap the
        embedding-lookup step for our embeddings, so this plugs straight into
        `Generator.prefill_forward_single_user_text`."""
        inputs_embeds = tokens if tokens.dim() == 3 else tokens.unsqueeze(0)
        S = inputs_embeds.shape[-2]
        # The base uses last_token_idx only for a seq-len assert; S already covers it.
        kwargs.pop("last_token_idx", None)
        dummy = torch.zeros(1, S, dtype=torch.long)
        out = list(super().prepare_inputs_prefill(dummy, **kwargs))
        out[0] = ttnn.from_torch(
            inputs_embeds.reshape(1, 1, S, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tuple(out)

    @torch.no_grad()
    def prefill_logits(self, inputs_embeds, page_table=None, kv_cache=None):
        """Run prefill on merged embeddings via the shared Generator single-user text
        path; return last-token logits (torch, vocab) and populate the KV cache for
        decoding. Pads the sequence to a multiple of 512, min 512 (the Blackhole
        prefill_len_cutoff / MLP reshape rule — see the comment on S_pad below); causal
        masking makes the trailing pad positions invisible to the last real token.

        ``page_table``/``kv_cache`` opt into paged KV. They default to None, which
        keeps the self-allocating single-shot behaviour. A serving stack that runs
        paged KV must pass them, because paged and non-paged decode dispatch to
        DIFFERENT SDPA kernels (paged_scaled_dot_product_attention_decode vs
        scaled_dot_product_attention_decode) and do not produce the same output."""
        S = inputs_embeds.shape[-2]
        last = S - 1
        # Always pad prefill to a multiple of 512 (the Blackhole prefill_len_cutoff), min 512.
        # tt_transformers' MLP reshapes prefill x to [1, S_pad//512, 512, -1] for S_pad >= 512, so
        # different padded lengths differ only in the batch dim -3 (512 -> [1,1,512,d], 1024 ->
        # [1,2,512,d]). A tt-metal program-cache collision (the prefill matmul hash doesn't cover
        # dim -3) then reuses the first bucket's program for the next: confirmed on device, a 512
        # prefill followed by a 1024 prefill TT_FATALs (1024 alone is fine). Real ASR prompts are
        # always <=512 tokens, so min-512 pins every request to the single [1,1,512,d] shape and
        # sidesteps the collision. Caps single-shot at max_seq_len (2048 -> ~150s); trailing pad is
        # causal-masked. See README "Known limitations" and docs/prefill_program_cache_collision_issue.md.
        S_pad = ((S + 511) // 512) * 512
        if S_pad != S:
            inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, S_pad - S))
        # Generator's single-user text prefill calls our embeds-aware prepare_inputs_prefill
        # + the shared ttnn_prefill_forward. Non-paged single-shot (page_table/kv_cache=None):
        # S_pad <= max_seq_len (2048) <= max_prefill_chunk_size, so it stays on the single-chunk
        # path (no paging required). It applies get_last_token=(last // 32) * 32 internally.
        # A paged prefill must see the user's page-table row trimmed to the blocks
        # the PADDED length covers, which is what Generator._get_prefill_user_page_table
        # does; handing it every block of the user's range makes the kernel attend
        # over KV pages beyond the prompt.
        prefill_page_table = page_table
        if page_table is not None and kv_cache is not None:
            num_blocks = num_blocks_in_seq(S_pad, get_block_size(kv_cache))
            prefill_page_table = page_table[0:1]
            if prefill_page_table.shape[1] < num_blocks:
                padding = torch.ones(1, num_blocks - prefill_page_table.shape[1], dtype=torch.int32) * -1
                prefill_page_table = torch.cat([prefill_page_table, padding], dim=1)
            prefill_page_table = prefill_page_table[:, :num_blocks]
        tt_logits = self.generator.prefill_forward_single_user_text(
            inputs_embeds,
            page_table=prefill_page_table,
            user_id=0,
            last_token_idx=last,
            kv_cache=kv_cache,
            batch_size=1,
        )
        tt_logits = ttnn.from_device(tt_logits)
        get_last = (last // 32) * 32
        full = self.process_output_prefill(tt_logits, last_token_idx=(last - get_last))
        return full.float(), S

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds,
        max_new_tokens=64,
        eos_id=None,
        repetition_penalty=1.0,
        prompt_ids=None,
        page_table=None,
        kv_cache=None,
    ):
        """Greedy decode.

        ``eos_id`` accepts a single id or a collection; it defaults to every stop
        id the checkpoint declares (see EOS_TOKEN_IDS).

        ``repetition_penalty`` mirrors the OpenAI/vLLM request parameter and must
        match whatever the serving path is asked for, or the two decode different
        token sequences from identical logits. ``prompt_ids`` are the prompt token
        ids: vLLM penalises prompt tokens as well as generated ones."""
        if eos_id is None:
            eos_ids = EOS_TOKEN_IDS
        elif isinstance(eos_id, int):
            eos_ids = (eos_id,)
        else:
            eos_ids = tuple(eos_id)
        seen = [] if prompt_ids is None else [int(i) for i in prompt_ids]
        logits, S = self.prefill_logits(inputs_embeds, page_table=page_table, kv_cache=kv_cache)
        nxt = int(apply_repetition_penalty(logits.reshape(-1), seen, repetition_penalty).argmax())
        out = [nxt]
        pos = S
        gen = self.generator
        # Host-side argmax greedy decode via the shared Generator with enable_trace=False.
        # Host argmax is faster here than an on-device ttnn.argmax over the 151936-wide vocab
        # (the wide reduction costs more than the logits host transfer), and a non-traced decode
        # stays stable across the mixed request shapes of a long-lived server (a persistent
        # decode trace did not — see README "Known limitations").
        while len(out) < max_new_tokens and nxt not in eos_ids:
            dl = gen.decode_forward(
                torch.tensor([[nxt]], dtype=torch.long),
                torch.tensor([pos]),
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=DECODE_TRACE,
                read_from_device=True,
            )
            dl = (dl[0] if isinstance(dl, tuple) else dl).squeeze().float().reshape(-1)
            nxt = int(apply_repetition_penalty(dl, seen + out, repetition_penalty).argmax())
            out.append(nxt)
            pos += 1
        if out and out[-1] in eos_ids:
            out = out[:-1]
        return out

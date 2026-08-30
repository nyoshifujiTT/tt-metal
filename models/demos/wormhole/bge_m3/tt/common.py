# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os

import ttnn


def resolve_model_name(model_name, model_location_generator=None, *, download_if_ci_v2=False, ci_v2_timeout_in_s=300):
    """Resolve a HF repo id to the checkpoint the caller should actually load.

    Order of precedence:

    1. ``HF_MODEL`` -- an explicit local snapshot directory. This is the
       convention the other demos use (e.g. gpt-oss, gemma) and lets demos and
       tests run offline, without reaching the Hugging Face Hub.
    2. ``model_location_generator`` -- the tt-metal fixture that resolves CIv2 /
       MLPerf cached checkpoints (and otherwise falls back to the repo id).
    3. The repo id itself, i.e. download from the Hub.

    ``download_if_ci_v2`` / ``ci_v2_timeout_in_s`` are forwarded to the fixture
    and default to the fixture's own defaults, so callers opt in to CIv2
    downloads explicitly instead of inheriting another model's policy.
    """
    hf_model = os.getenv("HF_MODEL")
    if hf_model:
        return hf_model
    if model_location_generator is None:
        return model_name
    return str(
        model_location_generator(
            model_name,
            download_if_ci_v2=download_if_ci_v2,
            ci_v2_timeout_in_s=ci_v2_timeout_in_s,
        )
    )


def create_tt_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat16,
    state_dict=None,
    hf_model_name="BAAI/bge-m3",
):
    """
    BGE-M3 version of create_tt_model that matches tt_transformers interface
    """
    from models.demos.wormhole.bge_m3.tt.model import BgeM3Model
    from models.demos.wormhole.bge_m3.tt.model_config import ModelArgs

    # Create BGE-M3 ModelArgs
    bge_m3_model_args = ModelArgs(
        mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        hf_model_name=hf_model_name,
    )

    if not state_dict:
        state_dict = bge_m3_model_args.load_state_dict()

    # Create BGE-M3 model
    model = BgeM3Model(
        args=bge_m3_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
    )

    return bge_m3_model_args, model, state_dict

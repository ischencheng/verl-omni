# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the SD3 FlowGRPO diffusers rollout adapter.

The adapter samples with the training adapter's forward, so the log-probs it
returns must equal what ``StableDiffusion3FlowGRPO`` recomputes from the same
trajectory, and a batch of requests must reproduce the per-request results.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import SD3Transformer2DModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "special_e2e"))

from build_sd3_tiny_random import ensure_tiny_sd3_checkpoint  # noqa: E402

from verl_omni.pipelines.model_base import DiffusersRolloutPipelineBase  # noqa: E402
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler  # noqa: E402
from verl_omni.pipelines.sd3_flow_grpo.diffusers_rollout_adapter import (  # noqa: E402
    StableDiffusion3DiffusersPipelineWithLogProb,
)
from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import StableDiffusion3FlowGRPO  # noqa: E402

PROMPTS = ["a red circle", "a blue square", "two green lines"]
SEEDS = [11, 12, 13]
STEPS = 4
SIZE = 64


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> str:
    return ensure_tiny_sd3_checkpoint(
        str(tmp_path_factory.mktemp("sd3_tiny")), hidden_size=16, seed=0, dtype=torch.float32
    )


@pytest.fixture(scope="module")
def pipeline(model_path) -> StableDiffusion3DiffusersPipelineWithLogProb:
    return StableDiffusion3DiffusersPipelineWithLogProb(model_path=model_path, dtype=torch.float32, device="cpu")


def _prompt(pipe, text: str) -> dict:
    ids = {
        "clip": pipe.tokenizer(text, add_special_tokens=True, truncation=True, max_length=77)["input_ids"],
        "t5": pipe.tokenizer_3(text, add_special_tokens=True, truncation=True, max_length=16)["input_ids"],
    }
    return {"prompt_ids": ids["t5"], "extra_args": {"extra_prompt_ids": ids}}


def _sampling(seed: int, **overrides) -> dict:
    params = {
        "height": SIZE,
        "width": SIZE,
        "num_inference_steps": STEPS,
        "max_sequence_length": 16,
        "guidance_scale": None,
        "noise_level": 0.7,
        "sde_window_size": 2,
        "sde_window_range": [0, 3],
        "sde_type": "sde",
        "logprobs": True,
        "output_type": "image",
        "seed": seed,
    }
    params.update(overrides)
    return params


def test_registered_under_the_vllm_omni_adapter_key():
    assert (
        DiffusersRolloutPipelineBase.get_class("StableDiffusion3Pipeline", "flow_grpo")
        is StableDiffusion3DiffusersPipelineWithLogProb
    )
    with pytest.raises(NotImplementedError, match="StableDiffusion3Pipeline"):
        DiffusersRolloutPipelineBase.get_class("StableDiffusion3Pipeline", "unknown_algo")


def test_trajectory_layout_and_seed_determinism(pipeline):
    first = pipeline.generate([_prompt(pipeline, PROMPTS[0])], [_sampling(SEEDS[0])])[0]
    again = pipeline.generate([_prompt(pipeline, PROMPTS[0])], [_sampling(SEEDS[0])])[0]
    other = pipeline.generate([_prompt(pipeline, PROMPTS[0])], [_sampling(SEEDS[1])])[0]

    latent_hw = SIZE // pipeline.vae_scale_factor
    assert first.all_latents.shape == (3, 16, latent_hw, latent_hw)
    assert first.all_log_probs.shape == (2,)
    assert first.all_timesteps.shape == (2,)
    with torch.no_grad():
        decoded = pipeline.vae.decode(
            first.latents.unsqueeze(0) / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor,
            return_dict=False,
        )[0][0]
    torch.testing.assert_close(first.image, decoded, rtol=0.0, atol=0.0)
    assert set(first.prompt_embeddings) >= {"prompt_embeds", "pooled_prompt_embeds", "prompt_embeds_mask"}
    torch.testing.assert_close(again.all_latents, first.all_latents, rtol=0.0, atol=0.0)
    torch.testing.assert_close(again.all_log_probs, first.all_log_probs, rtol=0.0, atol=0.0)
    assert not torch.equal(other.all_latents, first.all_latents)


def test_latent_output_decodes_only_for_a_requested_preview(pipeline):
    prompt = _prompt(pipeline, PROMPTS[0])
    latent_only = pipeline.generate([prompt], [_sampling(SEEDS[0], output_type="latent")])[0]
    with_preview = pipeline.generate(
        [prompt], [_sampling(SEEDS[0], output_type="latent", requested_outputs=["image_preview"])]
    )[0]

    assert latent_only.image is None
    assert with_preview.image is not None
    torch.testing.assert_close(with_preview.latents, latent_only.latents, rtol=0.0, atol=0.0)


def test_batch_matches_per_request(pipeline):
    prompts = [_prompt(pipeline, text) for text in PROMPTS]
    batched = pipeline.generate(prompts, [_sampling(seed) for seed in SEEDS])
    single = [pipeline.generate([p], [_sampling(seed)])[0] for p, seed in zip(prompts, SEEDS, strict=True)]

    for b, s in zip(batched, single, strict=True):
        torch.testing.assert_close(b.all_timesteps, s.all_timesteps, rtol=0.0, atol=0.0)
        torch.testing.assert_close(b.all_latents, s.all_latents, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b.all_log_probs, s.all_log_probs, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("guidance_scale", [None, 4.5])
def test_log_probs_match_training_recompute(pipeline, model_path, guidance_scale):
    prompt = _prompt(pipeline, PROMPTS[1])
    if guidance_scale is not None:
        prompt["extra_args"]["negative_extra_prompt_ids"] = _prompt(pipeline, "blurry")["extra_args"][
            "extra_prompt_ids"
        ]
    sample = pipeline.generate([prompt], [_sampling(SEEDS[1], guidance_scale=guidance_scale)])[0]

    module = SD3Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float32)
    scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(STEPS)
    model_config = SimpleNamespace(
        pipeline=SimpleNamespace(guidance_scale=guidance_scale),
        algo=SimpleNamespace(noise_level=0.7, sde_type="sde"),
    )
    latents = sample.all_latents.unsqueeze(0)
    timesteps = sample.all_timesteps.unsqueeze(0)
    embeds = sample.prompt_embeddings

    def inputs(step, prefix=""):
        return StableDiffusion3FlowGRPO.build_transformer_inputs(
            latents=latents[:, step],
            timesteps=timesteps[:, step],
            prompt_embeds=embeds[f"{prefix}prompt_embeds"].unsqueeze(0),
            pooled_prompt_embeds=embeds[f"{prefix}pooled_prompt_embeds"].unsqueeze(0),
        )

    recomputed = []
    with torch.no_grad():
        for step in range(timesteps.shape[1]):
            log_prob, *_ = StableDiffusion3FlowGRPO.forward_and_sample_previous_step(
                module,
                scheduler,
                model_config,
                inputs(step),
                inputs(step, "negative_") if guidance_scale is not None else None,
                {"all_latents": latents, "all_timesteps": timesteps},
                step,
            )
            recomputed.append(log_prob[0])
    torch.testing.assert_close(torch.stack(recomputed), sample.all_log_probs, rtol=0.0, atol=1e-5)


def test_rejects_unsupported_requests(pipeline):
    prompt = _prompt(pipeline, PROMPTS[0])
    with pytest.raises(ValueError, match="seed"):
        pipeline.generate([prompt], [_sampling(None)])
    with pytest.raises(ValueError, match="height"):
        pipeline.generate([prompt, prompt], [_sampling(1), _sampling(2, height=32)])
    with pytest.raises(ValueError, match="extra_prompt_ids"):
        pipeline.generate([{"prompt_ids": [1, 2]}], [_sampling(1)])
    with pytest.raises(ValueError, match="negative_extra_prompt_ids"):
        pipeline.generate([prompt], [_sampling(1, guidance_scale=4.5)])

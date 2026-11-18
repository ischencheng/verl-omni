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

"""SD3.5 diffusers rollout adapter for FlowGRPO."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor

from verl_omni.pipelines.model_base import DiffusersRolloutPipelineBase
from verl_omni.pipelines.request_batch import sample_per_sample_sde_windows
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec, requested_artifact_names
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.sd3_flow_grpo.common import (
    SD3_CLIP_TOKENS_KEY,
    SD3_ENCODER_TOKEN_KEYS,
    SD3_T5_TOKENS_KEY,
    SD3TokenIdPromptMixin,
)
from verl_omni.pipelines.sd3_flow_grpo.diffusers_training_adapter import (
    StableDiffusion3FlowGRPO,
    _calculate_shift,
    _sd3_image_seq_len,
)

__all__ = ["DiffusersRolloutSample", "StableDiffusion3DiffusersPipelineWithLogProb"]


@dataclass
class DiffusersRolloutSample:
    """One generated sample, in the per-sample layout the trainer receives from the vLLM-Omni adapter.

    ``W`` is the SDE window size.
    """

    image: torch.Tensor | None
    """Decoded image ``(C, H, W)`` in ``[-1, 1]``; ``None`` for latent output without an ``image_preview`` request."""
    latents: torch.Tensor
    """Final latents ``(C, h, w)``, fp32."""
    all_latents: torch.Tensor
    """SDE-window trajectory ``(W + 1, C, h, w)``, fp32."""
    all_log_probs: torch.Tensor | None
    """Per-step log-probs ``(W,)``, fp32; ``None`` when ``logprobs`` is off."""
    all_timesteps: torch.Tensor
    """Timesteps of the window steps ``(W,)``."""
    prompt_embeddings: dict[str, torch.Tensor | None] = field(default_factory=dict)
    """Per-sample prompt embeddings, keyed as in the vLLM-Omni adapter's ``prompt_embeddings``."""


def _flatten_sampling_params(sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in sampling_params.items() if key != "extra_args"}
    for key, value in (sampling_params.get("extra_args") or {}).items():
        if key in flat and flat[key] != value:
            raise ValueError(f"Conflicting diffusion sampling field {key!r} and extra_args.{key}")
        flat[key] = value
    return flat


def _batch_value(flat_params: list[dict[str, Any]], key: str, default: Any) -> Any:
    values = [params.get(key) for params in flat_params]
    values = [default if value is None else value for value in values]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"Batched diffusers rollout requests must share {key!r}, got {values}.")
    return values[0]


def _prompt_token_ids(prompt: Mapping[str, Any], key: str) -> dict[str, list[int]] | None:
    extra = (prompt.get("extra_args") or {}).get(key)
    if not extra:
        return None
    missing = [name for name in SD3_ENCODER_TOKEN_KEYS if name not in extra]
    if missing:
        raise ValueError(
            f"SD3 diffusers rollout requires {list(SD3_ENCODER_TOKEN_KEYS)} entries in '{key}' but {missing} "
            "are missing. Configure `actor_rollout_ref.model.extra_tokenizers` with these names."
        )
    return {
        name: extra[name].tolist() if isinstance(extra[name], torch.Tensor) else list(extra[name])
        for name in SD3_ENCODER_TOKEN_KEYS
    }


@DiffusersRolloutPipelineBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")
class StableDiffusion3DiffusersPipelineWithLogProb(SD3TokenIdPromptMixin):
    """SD3.5 FlowGRPO rollout on diffusers modules.

    Follows :class:`StableDiffusion3PipelineWithLogProb` (the vLLM-Omni adapter)
    step for step: token-id prompt encoding, per-request seeded latents and SDE
    windows, :class:`FlowMatchSDEDiscreteScheduler`, and the same trajectory
    layout. The denoiser is the diffusers ``SD3Transformer2DModel`` that the
    trainer optimizes, called through :meth:`StableDiffusion3FlowGRPO.forward`,
    so rollout and log-prob recomputation share one forward implementation.

    Args:
        model_path: Local diffusers checkpoint directory or Hub id.
        dtype: Dtype of the text encoders, VAE and transformer.
        device: Device the pipeline runs on.
    """

    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "CHW"),
        }
    )

    def __init__(self, *, model_path: str, dtype: torch.dtype, device: torch.device | str):
        local_files_only = os.path.exists(model_path)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=local_files_only
        )
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.text_encoder_2 = pipe.text_encoder_2.to(self.device)
        self.text_encoder_3 = pipe.text_encoder_3.to(self.device) if pipe.text_encoder_3 is not None else None
        self.vae = pipe.vae.to(self.device)
        self.transformer = pipe.transformer.to(self.device)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler", local_files_only=local_files_only
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.tokenizer_max_length = self.tokenizer.model_max_length

    def _model_dtype(self) -> torch.dtype:
        return self.dtype

    def _timesteps(self, num_inference_steps: int, sigmas: list[float] | None, height: int, width: int) -> torch.Tensor:
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None):
            scheduler_kwargs["mu"] = _calculate_shift(
                _sd3_image_seq_len(height, width),
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
        if sigmas is not None:
            self.scheduler.set_timesteps(sigmas=sigmas, **scheduler_kwargs)
        else:
            self.scheduler.set_timesteps(num_inference_steps, **scheduler_kwargs)
        return self.scheduler.timesteps

    def _encode(self, token_ids: list[dict[str, list[int]]], max_sequence_length: int):
        return self.encode_prompt_from_token_ids(
            clip_prompt_ids=[ids[SD3_CLIP_TOKENS_KEY] for ids in token_ids],
            t5_prompt_ids=[ids[SD3_T5_TOKENS_KEY] for ids in token_ids],
            max_sequence_length=max_sequence_length,
        )

    def _predict_noise(self, latents, timestep, prompt_embeds, pooled_prompt_embeds) -> torch.Tensor:
        model_inputs = StableDiffusion3FlowGRPO.build_transformer_inputs(
            latents=latents,
            timesteps=timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        return StableDiffusion3FlowGRPO.forward(self.transformer, None, model_inputs)

    @torch.no_grad()
    def generate(
        self, prompts: list[Mapping[str, Any]], sampling_params: list[Mapping[str, Any]]
    ) -> list[DiffusersRolloutSample]:
        """Generate one sample per request.

        Args:
            prompts: Diffusion prompt payloads (``OmniRolloutRequest.to_diffusion_prompt()``),
                carrying SD3 token ids under ``extra_args.extra_prompt_ids``.
            sampling_params: Agent-loop sampling dicts, one per prompt. Shape and SDE
                settings must match across the batch; ``seed`` is per request.

        Returns:
            One :class:`DiffusersRolloutSample` per request, in request order.
        """
        if len(prompts) != len(sampling_params) or not prompts:
            raise ValueError(f"Expected one sampling dict per prompt, got {len(prompts)} and {len(sampling_params)}.")
        flat_params = [_flatten_sampling_params(params) for params in sampling_params]
        height = _batch_value(flat_params, "height", None)
        width = _batch_value(flat_params, "width", None)
        if height is None or width is None:
            raise ValueError("SD3 diffusers rollout requires height and width in sampling params.")
        num_inference_steps = _batch_value(flat_params, "num_inference_steps", 28)
        sigmas = _batch_value(flat_params, "sigmas", None)
        max_sequence_length = _batch_value(flat_params, "max_sequence_length", 256)
        guidance_scale = float(_batch_value(flat_params, "guidance_scale", 0.0))
        noise_level = _batch_value(flat_params, "noise_level", 0.7)
        sde_window_size = _batch_value(flat_params, "sde_window_size", None)
        sde_window_range = tuple(_batch_value(flat_params, "sde_window_range", (0, 5)))
        sde_type = _batch_value(flat_params, "sde_type", "sde")
        logprobs = _batch_value(flat_params, "logprobs", True)
        output_type = _batch_value(flat_params, "output_type", "image")
        if output_type not in ("image", "latent", "both"):
            raise ValueError(f"SD3 output_type must be one of ['image', 'latent', 'both'], got {output_type!r}.")
        if any(params.get("num_outputs_per_prompt", 1) not in (None, 1) for params in flat_params):
            raise ValueError("SD3 diffusers rollout produces one output per request.")
        seeds = [params.get("seed") for params in flat_params]
        if any(seed is None for seed in seeds):
            raise ValueError("SD3 diffusers rollout requires a seed per request.")

        token_ids = [_prompt_token_ids(prompt, "extra_prompt_ids") for prompt in prompts]
        if any(ids is None for ids in token_ids):
            raise ValueError(
                "SD3 diffusers rollout requires per-text-encoder token ids under extra_args.extra_prompt_ids."
            )
        do_cfg = guidance_scale > 1
        prompt_embeds, pooled_prompt_embeds = self._encode(token_ids, max_sequence_length)
        negative_prompt_embeds = negative_pooled_prompt_embeds = None
        if do_cfg:
            negative_ids = [_prompt_token_ids(prompt, "negative_extra_prompt_ids") for prompt in prompts]
            if any(ids is None for ids in negative_ids):
                raise ValueError("SD3 diffusers rollout with CFG requires negative_extra_prompt_ids per request.")
            negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode(negative_ids, max_sequence_length)

        batch_size = len(prompts)
        generators = [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds]
        latent_shape = (
            batch_size,
            self.transformer.config.in_channels,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        latents = randn_tensor(latent_shape, generator=generators, device=self.device, dtype=self.dtype)
        timesteps = self._timesteps(num_inference_steps, sigmas, height, width)
        windows = sample_per_sample_sde_windows(
            sde_window_size=None if sde_window_size is None else int(sde_window_size),
            sde_window_range=sde_window_range,
            num_timesteps=len(timesteps),
            batch_size=batch_size,
            generator=generators,
            device=self.device,
        )
        if len({end - start for start, end in windows}) != 1:
            raise ValueError("Packed SDE windows must share the same size.")

        all_latents: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        all_log_probs: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        all_timesteps: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        self.scheduler.set_begin_index(0)
        for step, timestep_value in enumerate(timesteps):
            for row, (start, _) in enumerate(windows):
                if step == start:
                    all_latents[row].append(latents[row].detach().float().clone())
            levels = [float(noise_level) if start <= step < end else 0.0 for start, end in windows]
            cur_noise_level: float | torch.Tensor = (
                levels[0]
                if all(level == levels[0] for level in levels)
                else torch.tensor(levels, device=latents.device, dtype=torch.float32).view(
                    batch_size, *([1] * (latents.ndim - 1))
                )
            )
            timestep = timestep_value.expand(batch_size).to(device=self.device, dtype=self.dtype)
            model_latents = latents.to(self.dtype)
            noise_pred = self._predict_noise(model_latents, timestep, prompt_embeds, pooled_prompt_embeds)
            if do_cfg:
                negative_noise_pred = self._predict_noise(
                    model_latents, timestep, negative_prompt_embeds, negative_pooled_prompt_embeds
                )
                noise_pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=generators,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )
            for row, (start, end) in enumerate(windows):
                if start <= step < end:
                    all_latents[row].append(latents[row].detach().float().clone())
                    if log_prob is not None:
                        all_log_probs[row].append(log_prob[row])
                    all_timesteps[row].append(timestep_value)

        image = None
        requested = {
            name
            for params in flat_params
            for name in requested_artifact_names(params.get("requested_outputs"), context="SD3 diffusers rollout")
        }
        if output_type != "latent" or "image_preview" in requested:
            decode_latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor + self.vae.config.shift_factor
            image = self.vae.decode(decode_latents, return_dict=False)[0]

        embeddings = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": torch.ones(prompt_embeds.shape[:2], dtype=torch.int64, device=self.device),
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": None
            if negative_prompt_embeds is None
            else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.int64, device=self.device),
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
        return [
            DiffusersRolloutSample(
                image=None if image is None else image[row].cpu(),
                latents=latents[row].float().cpu(),
                all_latents=torch.stack(all_latents[row]).cpu(),
                all_log_probs=torch.stack(all_log_probs[row]).cpu() if all_log_probs[row] else None,
                all_timesteps=torch.stack(all_timesteps[row]).cpu(),
                prompt_embeddings={
                    name: None if value is None else value[row].cpu() for name, value in embeddings.items()
                },
            )
            for row in range(batch_size)
        ]

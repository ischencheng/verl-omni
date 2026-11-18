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
"""CPU tests for the diffusers rollout server on a tiny random SD3 checkpoint.

Necessity: ``rollout.name=diffusers`` must hand the agent loop the same
``DiffusionOutput`` fields as the vLLM-Omni SD3 adapter, survive the hybrid
sleep/wake cycle, and load actor weights from verl's bucketed sender for full,
merged and adapter-mode LoRA syncs.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "special_e2e"))

from build_sd3_tiny_random import ensure_tiny_sd3_checkpoint  # noqa: E402
from verl.workers.rollout.replica import RolloutMode  # noqa: E402
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer  # noqa: E402

from verl_omni.workers.rollout.diffusers_rollout import diffusers_server as server_module  # noqa: E402

HEIGHT = WIDTH = 64
STEPS = 4
WINDOW = 2
# Fields DiffusionStrategy.process_output returns for the SD3 vLLM-Omni adapter without CFG.
SD3_EXTRA_FIELDS = {
    "global_steps",
    "all_latents",
    "all_timesteps",
    "prompt_embeds",
    "prompt_embeds_mask",
    "pooled_prompt_embeds",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "negative_pooled_prompt_embeds",
    "media_kind",
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> str:
    return ensure_tiny_sd3_checkpoint(str(tmp_path_factory.mktemp("sd3")), dtype=torch.float32)


def _make_server(model_dir: str, rollout_mode=RolloutMode.HYBRID, replica_rank: int = 0):
    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        dtype="float32",
        free_cache_engine=True,
        external_lib=None,
        max_num_seqs=16,
    )
    model_config = SimpleNamespace(architecture="StableDiffusion3Pipeline", algorithm="flow_grpo", local_path=model_dir)
    with (
        patch.object(server_module, "omega_conf_to_dataclass", side_effect=lambda cfg, dataclass_type=None: cfg),
        patch.object(server_module, "get_device_name", return_value="cpu"),
    ):
        server = server_module.DiffusersRolloutServer(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            replica_rank=replica_rank,
            cuda_visible_devices="",
        )
    return server


def _request(server, seed: int, text: str, **overrides) -> dict:
    clip = server.pipeline.tokenizer
    t5 = server.pipeline.tokenizer_3
    sampling_params = {
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": STEPS,
        "guidance_scale": None,
        "max_sequence_length": 16,
        "noise_level": 0.7,
        "sde_window_size": WINDOW,
        "sde_window_range": [0, 3],
        "sde_type": "sde",
        "logprobs": True,
        "seed": seed,
        "global_steps": 3,
    }
    sampling_params.update(overrides)
    return {
        "prompt_ids": clip(text)["input_ids"],
        "sampling_params": sampling_params,
        "extra_prompt_ids": {
            "clip": clip(text, max_length=77, truncation=True)["input_ids"],
            "t5": t5(text, max_length=16, truncation=True)["input_ids"],
        },
    }


async def _launched(model_dir, **kwargs):
    server = _make_server(model_dir, **kwargs)
    await server.launch_server()
    return server


def test_generate_returns_the_vllm_omni_sd3_contract(model_dir):
    async def run():
        server = await _launched(model_dir)
        await server.set_global_steps(5)
        requests = [_request(server, 11, "a red cat"), _request(server, 12, "a blue dog")]
        outputs = await asyncio.gather(*[server.generate(request_id=f"r{i}", **req) for i, req in enumerate(requests)])
        solo = await server.generate(request_id="solo", **_request(server, 11, "a red cat"))
        return outputs, solo

    outputs, solo = asyncio.run(run())
    latent_hw = HEIGHT // 8
    for output in outputs:
        assert output.stop_reason == "completed"
        assert output.primary_artifact == "image_preview"
        assert output.preview_artifact == "image_preview"
        assert set(output.artifacts) == {"image_latent", "image_preview"}
        assert output.diffusion_output.dtype == torch.uint8
        # The tiny VAE has one up block, so the decoded size is its own, not 8x the latent.
        assert output.diffusion_output.shape[0] == 3
        assert output.artifacts["image_latent"].data.shape == (16, latent_hw, latent_hw)
        assert set(output.extra_fields) == SD3_EXTRA_FIELDS
        assert output.extra_fields["global_steps"] == 5
        assert output.extra_fields["media_kind"] == "image"
        assert output.extra_fields["all_latents"].shape == (WINDOW + 1, 16, latent_hw, latent_hw)
        assert output.extra_fields["all_timesteps"].shape == (WINDOW,)
        assert output.log_probs.shape == (WINDOW,)
        assert torch.isfinite(output.log_probs).all()
        assert output.extra_fields["negative_prompt_embeds"] is None
    assert not torch.equal(outputs[0].extra_fields["all_latents"], outputs[1].extra_fields["all_latents"])
    # A request's seed fixes its sample whether it ran batched or alone.
    torch.testing.assert_close(solo.extra_fields["all_latents"], outputs[0].extra_fields["all_latents"])
    torch.testing.assert_close(solo.log_probs, outputs[0].log_probs)


@pytest.mark.parametrize("output_type", ["latent", "both"])
def test_output_type_selects_primary_and_clean_latents(model_dir, output_type):
    async def run():
        server = await _launched(model_dir)
        request = _request(server, 3, "a cat", output_type=output_type, logprobs=False)
        return await server.generate(request_id="r", **request)

    output = asyncio.run(run())
    assert output.log_probs is None
    if output_type == "latent":
        assert output.primary_artifact == "image_latent"
        assert output.preview_artifact is None
        assert set(output.artifacts) == {"image_latent"}
        assert "latents_clean" not in output.extra_fields
    else:
        assert output.primary_artifact == "image_preview"
        torch.testing.assert_close(output.extra_fields["latents_clean"], output.artifacts["image_latent"].data)


def test_sleep_rejects_generation_until_wake_up(model_dir):
    async def run():
        server = await _launched(model_dir)
        await server.sleep()
        with pytest.raises(RuntimeError, match="asleep"):
            await server.generate(request_id="asleep", **_request(server, 1, "a cat"))
        await server.wake_up(tags=["kv_cache"])
        assert server._asleep
        await server.wake_up(tags=["weights"])
        return await server.generate(request_id="awake", **_request(server, 1, "a cat"))

    assert asyncio.run(run()).stop_reason == "completed"


def test_standalone_server_never_sleeps(model_dir):
    async def run():
        server = await _launched(model_dir, rollout_mode=RolloutMode.STANDALONE)
        await server.sleep()
        return server._asleep

    assert asyncio.run(run()) is False


def test_abort_returns_pending_requests_as_aborted(model_dir):
    async def run():
        server = await _launched(model_dir)
        async with server._lock:  # hold the pipeline so the request stays pending
            task = asyncio.create_task(server.generate(request_id="pending", **_request(server, 1, "a cat")))
            await asyncio.sleep(0.05)
            result = await server.abort_all_requests()
        aborted = await task
        await server.resume_generation()
        resumed = await server.generate(request_id="after", **_request(server, 1, "a cat"))
        return result, aborted, resumed

    result, aborted, resumed = asyncio.run(run())
    assert result["request_ids"] == ["pending"]
    assert aborted.stop_reason == "aborted"
    assert resumed.stop_reason == "completed"


def _sync(server, monkeypatch, named_tensors, **receiver_kwargs):
    """Feed one sync through the server's receiver, reusing one bucket buffer like the IPC wire."""
    seen = {}
    tensors = list(named_tensors)

    class Receiver:
        def __init__(self, *, zmq_handle, device, use_shm):
            seen.update(zmq_handle=zmq_handle, device=device, use_shm=use_shm)

        def receive_weights(self, on_bucket_received):
            for index, (name, tensor) in enumerate(tensors):
                buffer = tensor.clone()
                on_bucket_received([(name, buffer)], index == len(tensors) - 1)
                buffer.fill_(-99)  # a later bucket overwrites the shared buffer

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", Receiver)
    server.update_weights_from_ipc(use_shm=False, **receiver_kwargs)
    return seen


def test_full_weight_sync_loads_every_transformer_tensor(model_dir, monkeypatch):
    monkeypatch.setenv("VERL_RAY_JOB_ID", "cpu-test-full")
    server = asyncio.run(_launched(model_dir, replica_rank=3))
    transformer = server.pipeline.transformer
    expected = {name: torch.randn_like(param) for name, param in transformer.state_dict().items()}
    seen = _sync(server, monkeypatch, [(f"transformer.{name}", tensor) for name, tensor in expected.items()])
    # Same socket as ServerAdapter.zmq_handle for replica 3, rollout rank 0.
    assert seen["zmq_handle"] == "ipc:///tmp/rl-colocate-zmq-cpu-test-full-replica-3-rank-0.sock"
    for name, tensor in transformer.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_adapter_sync_injects_then_replaces_lora(model_dir, monkeypatch):
    monkeypatch.setenv("VERL_RAY_JOB_ID", "cpu-test-lora")
    server = asyncio.run(_launched(model_dir))
    transformer = server.pipeline.transformer
    peft_config = {"r": 2, "lora_alpha": 4, "target_modules": ["to_q", "to_k"], "peft_type": "LORA"}
    reference = {name: tensor.clone() for name, tensor in transformer.state_dict().items()}

    for step in range(2):
        adapter = {}
        for name, module in transformer.named_modules():
            if name.split(".")[-1] in ("to_q", "to_k"):
                linear = module.base_layer if hasattr(module, "base_layer") else module
                adapter[f"transformer.{name}.lora_A.weight"] = torch.full((2, linear.in_features), 0.1 * (step + 1))
                adapter[f"transformer.{name}.lora_B.weight"] = torch.full((linear.out_features, 2), 0.2)
        seen = _sync(
            server,
            monkeypatch,
            adapter.items(),
            peft_config=peft_config,
            base_sync_done=True,
            zmq_update_id=f"step-{step}-seq-{step}",
        )
        assert seen["zmq_handle"].endswith(f"-replica-0-rank-0-update-step-{step}-seq-{step}.sock")
        q_proj = transformer.transformer_blocks[0].attn.to_q  # PEFT swaps the module on injection
        lora_a, lora_b = q_proj.lora_A["default"].weight, q_proj.lora_B["default"].weight
        torch.testing.assert_close(lora_a, torch.full_like(lora_a, 0.1 * (step + 1)))
        torch.testing.assert_close(lora_b, torch.full_like(lora_b, 0.2))

    # A later full sync still reaches the wrapped base weights.
    expected = {name: tensor + 1 for name, tensor in reference.items()}
    _sync(server, monkeypatch, [(f"transformer.{name}", tensor) for name, tensor in expected.items()])
    expected_q = expected["transformer_blocks.0.attn.to_q.weight"]
    torch.testing.assert_close(q_proj.base_layer.weight, expected_q, rtol=0, atol=0)


def test_adapter_sync_rejects_unknown_lora_tensors(model_dir, monkeypatch):
    monkeypatch.setenv("VERL_RAY_JOB_ID", "cpu-test-lora-unknown")
    server = asyncio.run(_launched(model_dir))
    peft_config = {"r": 2, "lora_alpha": 4, "target_modules": ["to_q"], "peft_type": "LORA"}
    bogus = [("transformer.transformer_blocks.0.attn.to_v.lora_A.weight", torch.zeros(2, 8))]
    with pytest.raises(ValueError, match="does not hold"):
        _sync(server, monkeypatch, bogus, peft_config=peft_config, base_sync_done=True)

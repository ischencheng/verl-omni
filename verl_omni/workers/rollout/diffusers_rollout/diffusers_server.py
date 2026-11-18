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
"""Diffusers rollout backend (``actor_rollout_ref.rollout.name=diffusers``).

One server actor per replica runs the registered
:class:`~verl_omni.pipelines.model_base.DiffusersRolloutPipelineBase` pipeline
in its own process on the replica's GPU. It serves the same ``generate`` and
lifecycle calls as the vLLM-Omni server, returns the same
:class:`~verl_omni.workers.rollout.replica.DiffusionOutput` fields, and
receives actor weights over verl's bucketed ZMQ/IPC wire.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import ray
import torch
from verl.plugin.platform import get_platform
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_resource_name, get_torch_device, get_visible_devices_keyword
from verl.utils.import_utils import import_external_libs
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.vllm_rollout import ServerAdapter

from verl_omni.pipelines.diffusion_rollout_output import quantize_pixels
from verl_omni.pipelines.model_base import DiffusersRolloutPipelineBase
from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, requested_artifact_names, validate_artifacts
from verl_omni.pipelines.rollout_request import OmniRolloutRequest
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.base import get_rollout_world_size
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import make_update_zmq_handle

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

__all__ = ["DiffusersReplica", "DiffusersRolloutServer", "DiffusersServerAdapter", "build_diffusion_output"]

_TRANSFORMER_PREFIX = "transformer."
# Sampling fields that may differ inside one batched pipeline call.
_PER_REQUEST_FIELDS = frozenset({"seed", "global_steps"})
# LoraConfig fields that shape the adapter forward; training-only fields (dropout, init) are dropped.
_LORA_FORWARD_FIELDS = (
    "r",
    "lora_alpha",
    "target_modules",
    "exclude_modules",
    "use_rslora",
    "rank_pattern",
    "alpha_pattern",
    "fan_in_fan_out",
    "bias",
    "layers_to_transform",
    "layers_pattern",
)


def _flat_sampling_params(sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in sampling_params.items() if key != "extra_args"}
    flat.update(sampling_params.get("extra_args") or {})
    return flat


def _batch_key(sampling_params: Mapping[str, Any]) -> str:
    flat = _flat_sampling_params(sampling_params)
    return repr(sorted((key, repr(value)) for key, value in flat.items() if key not in _PER_REQUEST_FIELDS))


def build_diffusion_output(
    sample: Any,
    sampling_params: Mapping[str, Any],
    *,
    global_steps: Optional[int],
    io_spec: Any,
    context: str,
) -> DiffusionOutput:
    """Convert one :class:`DiffusersRolloutSample` to the trainer-facing output.

    Mirrors ``DiffusionStrategy.process_output`` for visual adapters: a native
    ``image_latent`` artifact, a uint8 ``image_preview`` when decoded pixels exist,
    the trajectory and prompt embeddings in ``extra_fields``, and ``latents_clean``
    for ``output_type="both"``.

    Args:
        sample: Per-sample pipeline output (``image`` ``(C, H, W)`` in ``[-1, 1]`` or None,
            ``latents`` ``(C, h, w)``, ``all_latents``, ``all_log_probs``, ``all_timesteps``,
            ``prompt_embeddings``).
        sampling_params: The request's public sampling dict.
        global_steps: Weight version stamped on the output.
        io_spec: The pipeline's ``DiffusionIOSpec`` declaration, or None.
        context: Pipeline/request description used in validation errors.

    Returns:
        The rollout output the diffusion agent loop consumes.
    """
    flat = _flat_sampling_params(sampling_params)
    output_type = flat.get("output_type") or "image"
    specs = {"image_latent": MediaSpec("image", "latent", "CHW")}
    items = [("image_latent", MediaArtifact(specs["image_latent"], sample.latents))]
    if sample.image is not None:
        specs["image_preview"] = MediaSpec("image", "decoded", "CHW")
        pixels = quantize_pixels(sample.image, "minus_one_one", context=context)
        items.append(("image_preview", MediaArtifact(specs["image_preview"], pixels)))
    primary = "image_latent" if output_type == "latent" else "image_preview"
    if primary not in specs:
        raise ValueError(f"{context}: output_type={output_type!r} needs decoded pixels, but the pipeline returned none")
    requested = requested_artifact_names(flat.get("requested_outputs"), context=context)
    artifacts = validate_artifacts(items, specs, primary=primary, context=context, requested=requested)
    if io_spec is not None:
        for name, artifact in artifacts.items():
            if io_spec.artifacts.get(name) != artifact.spec:
                raise ValueError(f"{context}: artifact={name!r} does not match the pipeline declaration")

    fields: dict[str, Any] = {
        "global_steps": global_steps,
        "all_latents": sample.all_latents,
        "all_timesteps": sample.all_timesteps,
    }
    for key, value in sample.prompt_embeddings.items():
        if key in fields:
            raise ValueError(f"Duplicate rollout metadata field: {key}")
        fields[key] = value
    if output_type == "both":
        fields["latents_clean"] = sample.latents.float()
    fields["media_kind"] = "image"
    return DiffusionOutput(
        artifacts=artifacts,
        primary_artifact=primary,
        preview_artifact="image_preview" if "image_preview" in artifacts else None,
        diffusion_output=artifacts[primary].data,
        log_probs=sample.all_log_probs if flat.get("logprobs", False) else None,
        stop_reason="completed",
        num_preempted=None,
        extra_fields=fields,
    )


def _aborted_output(global_steps: Optional[int]) -> DiffusionOutput:
    return DiffusionOutput(
        diffusion_output=torch.empty(0, dtype=torch.uint8),
        log_probs=None,
        stop_reason="aborted",
        num_preempted=None,
        extra_fields={"global_steps": global_steps},
    )


@dataclass(eq=False)
class _PendingRequest:
    request_id: str
    prompt: dict[str, Any]
    sampling_params: dict[str, Any]
    future: asyncio.Future


class DiffusersRolloutServer:
    """Rollout server for one diffusers replica (one GPU).

    Args:
        config: Rollout config (``DiffusionRolloutConfig`` fields).
        model_config: Actor model config (``DiffusionModelConfig`` fields).
        rollout_mode: Hybrid, colocated or standalone.
        replica_rank: Replica index; part of the weight-sync socket name.
        cuda_visible_devices: Device list of the replica's actor worker.
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        replica_rank: int,
        cuda_visible_devices: str,
    ):
        if cuda_visible_devices:
            os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        if ray.is_initialized():
            os.environ["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()
        self.config: DiffusionRolloutConfig = omega_conf_to_dataclass(config, dataclass_type=DiffusionRolloutConfig)
        self.model_config: DiffusionModelConfig = omega_conf_to_dataclass(
            model_config, dataclass_type=DiffusionModelConfig
        )
        if get_rollout_world_size(self.config) != 1:
            raise ValueError("The diffusers rollout runs one GPU per replica; set TP, DP, PP, Ulysses and ring to 1.")
        self.rollout_mode = rollout_mode
        self.replica_rank = replica_rank
        self.global_steps: Optional[int] = None
        device_name = get_device_name()
        self.device = torch.device(device_name) if device_name == "cpu" else torch.device(device_name, 0)
        self.pipeline = None
        self._asleep = False
        self._lock = asyncio.Lock()  # guards the pipeline: generation, sleep/wake and weight loads
        self._pending: list[_PendingRequest] = []
        self._wakeup = asyncio.Event()
        self._admission_open = asyncio.Event()
        self._admission_open.set()
        self._reject_requests = False
        self._batch_task: Optional[asyncio.Task] = None
        self._adapter_injected = False

    async def launch_server(self) -> None:
        """Load the registered pipeline and start the batching loop."""
        import_external_libs(self.config.external_lib)
        pipeline_cls = DiffusersRolloutPipelineBase.get_class(
            self.model_config.architecture, self.model_config.algorithm
        )
        dtype = getattr(torch, self.config.dtype)
        self.pipeline = await asyncio.to_thread(
            pipeline_cls, model_path=self.model_config.local_path, dtype=dtype, device=self.device
        )
        self._batch_task = asyncio.get_running_loop().create_task(self._batch_loop())

    def get_server_address(self) -> str:
        """Return a unique router key; the backend has no HTTP endpoint."""
        return f"diffusers://replica-{self.replica_rank}"

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        priority: int = 0,
        session_id: Optional[str] = None,
    ) -> DiffusionOutput:
        """Generate one sample; concurrent compatible requests run as one pipeline batch."""
        if image_data is not None or video_data is not None or audio_data is not None:
            raise NotImplementedError("The diffusers rollout does not take condition media yet.")
        if priority != 0:
            raise ValueError("The diffusers rollout does not support nonzero request priority.")
        request = OmniRolloutRequest.from_generate_kwargs(
            prompt_ids=normalize_token_ids(prompt_ids),
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            extra_prompt_ids=extra_prompt_ids,
            negative_extra_prompt_ids=negative_extra_prompt_ids,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if self._reject_requests:
            return _aborted_output(self.global_steps)
        await self._admission_open.wait()
        future = asyncio.get_running_loop().create_future()
        self._pending.append(_PendingRequest(request_id, dict(request.to_diffusion_prompt()), sampling_params, future))
        self._wakeup.set()
        return await future

    async def _batch_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            while self._pending:
                async with self._lock:
                    if not self._pending:  # aborted while waiting for the lock
                        break
                    key = _batch_key(self._pending[0].sampling_params)
                    batch = [req for req in self._pending if _batch_key(req.sampling_params) == key]
                    batch = batch[: self.config.max_num_seqs]
                    self._pending = [req for req in self._pending if all(req is not taken for taken in batch)]
                    if self._asleep:
                        error = RuntimeError("diffusers rollout received a request while asleep")
                        for req in batch:
                            req.future.set_exception(error)
                        continue
                    try:
                        outputs = await asyncio.to_thread(self._run_batch, batch)
                    except Exception as error:  # noqa: BLE001 -- delivered to every caller of this batch
                        for req in batch:
                            if not req.future.done():
                                req.future.set_exception(error)
                        continue
                for req, output in zip(batch, outputs, strict=True):
                    if not req.future.done():
                        req.future.set_result(output)

    def _run_batch(self, batch: list[_PendingRequest]) -> list[DiffusionOutput]:
        samples = self.pipeline.generate([req.prompt for req in batch], [req.sampling_params for req in batch])
        pipeline = f"{self.model_config.architecture}/{self.model_config.algorithm}"
        io_spec = getattr(self.pipeline, "diffusion_io_spec", None)
        return [
            build_diffusion_output(
                sample,
                req.sampling_params,
                global_steps=self.global_steps,
                io_spec=io_spec,
                context=f"pipeline={pipeline}, request_id={req.request_id}",
            )
            for req, sample in zip(batch, samples, strict=True)
        ]

    async def abort_all_requests(self, reject_request: bool = False, **kwargs) -> dict[str, Any]:
        """Abort requests that have not started; the running batch finishes."""
        aborted = self._pending
        self._pending = []
        self._admission_open.clear()
        self._reject_requests = reject_request
        for req in aborted:
            if not req.future.done():
                req.future.set_result(_aborted_output(self.global_steps))
        return {"aborted_count": len(aborted), "request_ids": [req.request_id for req in aborted]}

    async def resume_generation(self) -> None:
        """Re-open admission after ``abort_all_requests``."""
        self._reject_requests = False
        self._admission_open.set()

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort one request if it has not started."""
        for req in list(self._pending):
            if req.request_id == request_id:
                self._pending.remove(req)
                req.future.set_result(_aborted_output(self.global_steps))
                return {"aborted": True, "request_id": request_id}
        return {"aborted": False, "request_id": request_id}

    async def wait_for_requests_to_drain(self) -> None:
        """Wait until no request is pending or running."""
        while self._pending:
            await asyncio.sleep(0.01)
        async with self._lock:
            pass

    # -----------------------------------------------------------------------
    # Memory lifecycle
    # -----------------------------------------------------------------------

    def _move_pipeline(self, device: torch.device) -> None:
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae", "transformer"):
            module = getattr(self.pipeline, name, None)
            if module is not None:
                module.to(device)
        if device.type == "cpu" and self.device.type != "cpu":
            get_torch_device().empty_cache()

    async def sleep(self) -> None:
        """Offload the pipeline to host memory between rollouts (hybrid and colocated modes)."""
        if not self.config.free_cache_engine or self.rollout_mode == RolloutMode.STANDALONE:
            return
        await self.wait_for_requests_to_drain()
        async with self._lock:
            if not self._asleep:
                await asyncio.to_thread(self._move_pipeline, torch.device("cpu"))
                self._asleep = True

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Move the pipeline back to the GPU; only the ``weights`` tag holds memory here."""
        if self.rollout_mode == RolloutMode.STANDALONE:
            return
        if tags is not None and "weights" not in tags:
            return
        async with self._lock:
            if self._asleep:
                await asyncio.to_thread(self._move_pipeline, self.device)
                self._asleep = False

    async def clear_kv_cache(self) -> None:
        """No KV cache in a diffusion pipeline."""

    async def release_kv_cache(self) -> None:
        """No KV cache in a diffusion pipeline."""

    async def resume_kv_cache(self) -> None:
        """No KV cache in a diffusion pipeline."""

    async def set_global_steps(self, global_steps: int) -> None:
        """Stamp the weight version reported on later outputs."""
        self.global_steps = global_steps

    async def start_profile(self, **kwargs) -> None:
        """Profiling is not wired for this backend."""

    async def stop_profile(self) -> None:
        """Profiling is not wired for this backend."""

    # -----------------------------------------------------------------------
    # Weight sync
    # -----------------------------------------------------------------------

    async def collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        require_ack: bool = False,
    ) -> list[Any]:
        """Run a worker-side method; only the weight-sync receiver exists on this backend."""
        if method != "update_weights_from_ipc":
            raise NotImplementedError(f"diffusers rollout does not implement collective_rpc({method!r})")
        async with self._lock:
            await asyncio.to_thread(self.update_weights_from_ipc, *args, **(kwargs or {}))
        return [None]

    def _zmq_handle(self, zmq_update_id: Optional[str]) -> str:
        # Same name as ServerAdapter.zmq_handle for rollout rank 0 of this replica.
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-0.sock"
        return make_update_zmq_handle(handle, zmq_update_id) if zmq_update_id is not None else handle

    def update_weights_from_ipc(
        self,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        zmq_update_id: Optional[str] = None,
        delta_flush: Optional[bool] = None,
    ) -> None:
        """Receive one weight sync from the colocated actor into the pipeline transformer.

        Full and merged-LoRA syncs copy each tensor into the matching parameter.
        Adapter syncs (``peft_config`` with ``base_sync_done``) collect the LoRA
        tensors and load them into a PEFT adapter injected on the first sync.
        """
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if delta_flush:
            raise NotImplementedError("The diffusers rollout does not apply delta_sharded weight syncs.")
        if self._asleep:
            raise RuntimeError("Weight sync reached the diffusers rollout while its pipeline is offloaded.")
        receiver = BucketedWeightReceiver(
            zmq_handle=self._zmq_handle(zmq_update_id), device=self.device, use_shm=use_shm
        )
        transformer = self.pipeline.transformer
        if peft_config is not None and base_sync_done:
            adapter_state: dict[str, torch.Tensor] = {}

            def on_bucket(weights, is_last=False):
                for name, tensor in weights:
                    adapter_state[_strip_transformer_prefix(name)] = tensor.clone()

            receiver.receive_weights(on_bucket_received=on_bucket)
            self._load_adapter(transformer, adapter_state, peft_config)
            return

        params = dict(transformer.named_parameters())
        params.update(dict(transformer.named_buffers()))

        def on_bucket(weights, is_last=False):
            with torch.no_grad():
                for name, tensor in weights:
                    target = _resolve_param(params, _strip_transformer_prefix(name))
                    if target.shape != tensor.shape:
                        raise ValueError(
                            f"Weight {name!r}: expected shape {tuple(target.shape)}, got {tuple(tensor.shape)}"
                        )
                    target.copy_(tensor)

        receiver.receive_weights(on_bucket_received=on_bucket)

    def _load_adapter(self, transformer, adapter_state: dict[str, torch.Tensor], peft_config: dict) -> None:
        from peft import LoraConfig, set_peft_model_state_dict

        if not self._adapter_injected:
            if peft_config.get("use_dora"):
                raise NotImplementedError("The diffusers rollout loads plain LoRA adapters; DoRA is not wired.")
            config = LoraConfig(**{key: peft_config[key] for key in _LORA_FORWARD_FIELDS if key in peft_config})
            transformer.add_adapter(config, adapter_name="default")
            transformer.eval()  # injected modules start in training mode
            self._adapter_injected = True
        adapter_state = {re.sub(r"\.default(?=\.)", "", name): tensor for name, tensor in adapter_state.items()}
        result = set_peft_model_state_dict(transformer, adapter_state, adapter_name="default")
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise ValueError(f"LoRA sync carried tensors the rollout adapter does not hold: {unexpected[:5]}")


def _strip_transformer_prefix(name: str) -> str:
    return name[len(_TRANSFORMER_PREFIX) :] if name.startswith(_TRANSFORMER_PREFIX) else name


def _resolve_param(params: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name in params:
        return params[name]
    # A PEFT-injected layer keeps the base weight under ``base_layer``.
    module, _, leaf = name.rpartition(".")
    wrapped = f"{module}.base_layer.{leaf}"
    if wrapped in params:
        return params[wrapped]
    raise KeyError(f"Weight sync tensor {name!r} has no matching parameter in the rollout transformer.")


class DiffusersServerAdapter(ServerAdapter):
    """Actor-side handle: verl's bucketed sender plus the replica-rank socket naming."""

    def __init__(self, config, model_config, device_mesh, replica_rank: int = -1):
        super().__init__(config, model_config, device_mesh, replica_rank=replica_rank)
        if get_rollout_world_size(self.config) != 1:
            raise ValueError("The diffusers rollout runs one GPU per replica; set TP, DP, PP, Ulysses and ring to 1.")

    def _get_server_name_prefix(self) -> str:
        return "diffusers_"


class DiffusersReplica(RolloutReplica):
    """One diffusers server actor per replica, placed on its actor worker's GPU."""

    async def launch_servers(self) -> None:
        """Start the server actor next to the replica's worker and load the pipeline."""
        if self.world_size != 1:
            raise ValueError("The diffusers rollout runs one GPU per replica; set TP, DP and PP to 1.")
        worker = self.workers[0]
        node_id, device_id = await worker.__ray_call__.remote(
            lambda self: (
                ray.get_runtime_context().get_node_id(),
                ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
            )
        )
        if self.is_reward_model:
            name = f"diffusers_server_reward_{self.replica_rank}_0{self.name_suffix}"
        elif self.is_teacher_model:
            name = f"diffusers_server_teacher_{self.replica_rank}_0{self.name_suffix}"
        else:
            name = f"diffusers_server_{self.replica_rank}_0{self.name_suffix}"
        env_vars = {**{var: "1" for var in get_platform().ray_noset_envvars()}, **get_platform().rollout_env_vars()}
        server = (
            ray.remote(DiffusersRolloutServer)
            .options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
                max_concurrency=self.config.ray_actor_max_concurrency,
            )
            .remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                replica_rank=self.replica_rank,
                cuda_visible_devices=str(device_id),
            )
        )
        await server.launch_server.remote()
        self.servers = [server]
        self._server_handle = server
        self._server_address = await server.get_server_address.remote()

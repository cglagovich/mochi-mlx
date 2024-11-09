import json
import os
import random
from functools import partial
from typing import Dict, List

from safetensors.torch import load_file
import numpy as np
import torch
# import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import yaml
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch import nn
# from torch.distributed.fsdp import (
#     BackwardPrefetch,
#     MixedPrecision,
#     ShardingStrategy,
# )
# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
# )
# from torch.distributed.fsdp.wrap import (
#     lambda_auto_wrap_policy,
#     transformer_auto_wrap_policy,
# )
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

import mochi_preview.dit.joint_model.context_parallel as cp
import mochi_preview.vae.cp_conv as cp_conv
from mochi_preview.utils import Timer
from mochi_preview.vae.model import Decoder

T5_MODEL = "google/t5-v1_1-xxl"
MAX_T5_TOKEN_LENGTH = 256

class T5_Tokenizer:
    """Wrapper around Hugging Face tokenizer for T5

    Args:
        model_name(str): Name of tokenizer to load.
    """

    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(T5_MODEL, legacy=False)

    def __call__(self, prompt, padding, truncation, return_tensors, max_length=None):
        """
        Args:
            prompt (str): The input text to tokenize.
            padding (str): The padding strategy.
            truncation (bool): Flag indicating whether to truncate the tokens.
            return_tensors (str): Flag indicating whether to return tensors.
            max_length (int): The max length of the tokens.
        """
        assert (
            not max_length or max_length == MAX_T5_TOKEN_LENGTH
        ), f"Max length must be {MAX_T5_TOKEN_LENGTH} for T5."

        tokenized_output = self.tokenizer(
            prompt,
            padding=padding,
            max_length=MAX_T5_TOKEN_LENGTH,  # Max token length for T5 is set here.
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )

        return tokenized_output


def unnormalize_latents(
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Unnormalize latents. Useful for decoding DiT samples.

    Args:
        z (torch.Tensor): [B, C_z, T_z, H_z, W_z], float

    Returns:
        torch.Tensor: [B, C_z, T_z, H_z, W_z], float
    """
    mean = mean[:, None, None, None]
    std = std[:, None, None, None]

    assert z.ndim == 5
    assert z.size(1) == mean.size(0) == std.size(0)
    return z * std.to(z) + mean.to(z)


# def setup_fsdp_sync(model, device_id, *, param_dtype, auto_wrap_policy) -> FSDP:
#     model = FSDP(
#         model,
#         sharding_strategy=ShardingStrategy.FULL_SHARD,
#         mixed_precision=MixedPrecision(
#             param_dtype=param_dtype,
#             reduce_dtype=torch.float32,
#             buffer_dtype=torch.float32,
#         ),
#         auto_wrap_policy=auto_wrap_policy,
#         backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
#         limit_all_gathers=True,
#         device_id=device_id,
#         sync_module_states=True,
#         use_orig_params=True,
#     )
#     # torch.cuda.synchronize()
#     return model

def setup_fsdp_sync(model, *args, **kwargs):
    return model


def compute_packed_indices(
    N: int,
    text_mask: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        N: Number of visual tokens.
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    assert N > 0 and len(text_mask) == 1
    text_mask = text_mask[0]

    mask = F.pad(text_mask, (N, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(
        mask.flatten(), as_tuple=False
    ).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * N  # At least (B * N,)
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens,
        "max_seqlen_in_batch_kv": max_seqlen_in_batch,
        "valid_token_indices_kv": valid_token_indices,
    }


def shift_sigma(
    sigma: np.ndarray,
    shift: float,
):
    """Shift noise standard deviation toward higher values.

    Useful for training a model at high resolutions,
    or sampling more finely at high noise levels.

    Equivalent to:
        sigma_shift = shift / (shift + 1 / sigma - 1)
    except for sigma = 0.

    Args:
        sigma: noise standard deviation in [0, 1]
        shift: shift factor >= 1.
               For shift > 1, shifts sigma to higher values.
               For shift = 1, identity function.
    """
    return shift * sigma / (shift * sigma + 1 - sigma)


class T2VSynthMochiModel:
    def __init__(
        self,
        *,
        device_id: int,
        world_size: int,
        local_rank: int,
        vae_stats_path: str,
        vae_checkpoint_path: str,
        dit_config_path: str,
        dit_checkpoint_path: str,
    ):
        super().__init__()
        t = Timer()
        self.device = torch.device(device_id)
        if not self.device.type == "cuda":
            self.device = torch.device("cpu")
        if world_size > 1:
            assert False
            # os.environ["MASTER_ADDR"] = "127.0.0.1"
            # os.environ["MASTER_PORT"] = "29500"
            # with t("init_process_group (can take 20-30 seconds)"):
            #     dist.init_process_group(
            #         "nccl",
            #         rank=local_rank,
            #         world_size=world_size,
            #         device_id=self.device,  # force non-lazy init
            #     )
            #     # get the default PG
            #     pg = dist.group.WORLD
            #     cp.set_cp_group(pg, list(range(world_size)), local_rank)

        self.t5_tokenizer = T5_Tokenizer()

        with t("load_text_encs"):
            t5_enc = T5EncoderModel.from_pretrained(T5_MODEL).eval().to(self.device)
            # self.t5_enc = (
            #     setup_fsdp_sync(
            #         t5_enc,
            #         device_id=device_id,
            #         param_dtype=torch.float32,
            #         auto_wrap_policy=partial(
            #             transformer_auto_wrap_policy,
            #             transformer_layer_cls={
            #                 T5Block,
            #             },
            #         ),
            #     )
            #     if world_size > 1
            #     else t5_enc.to(self.device)
            # )
            self.t5_enc = t5_enc
            self.t5_enc.to(self.device)
            self.t5_enc.eval()

        with t("load_vae"):
            self.decoder = Decoder(
                out_channels=3,
                base_channels=128,
                channel_multipliers=[1, 2, 4, 6],
                temporal_expansions=[1, 2, 3],
                spatial_expansions=[2, 2, 2],
                num_res_blocks=[3, 3, 4, 6, 3],
                latent_dim=12,
                has_attention=[False, False, False, False, False],
                padding_mode="replicate",
                output_norm=False,
                nonlinearity="silu",
                output_nonlinearity="silu",
                causal=True,
            )
            decoder_sd = load_file(vae_checkpoint_path)
            if not os.environ.get("NO_LOAD", False):
                self.decoder.load_state_dict(decoder_sd, strict=True)
            self.decoder.eval().to(self.device)

        with t("construct_dit"):
            with open(dit_config_path, "r") as f:
                from mochi_preview.dit.joint_model.asymm_models_joint import (
                    AsymmDiTJoint,
                )
                model: nn.Module = torch.nn.utils.skip_init(
                    AsymmDiTJoint,
                    depth=48,
                    patch_size=2,
                    num_heads=24,
                    hidden_size_x=3072,
                    hidden_size_y=1536,
                    mlp_ratio_x=4.0,
                    mlp_ratio_y=4.0,
                    in_channels=12,
                    qk_norm=True,
                    qkv_bias=False,
                    out_bias=True,
                    patch_embed_bias=True,
                    timestep_mlp_bias=True,
                    timestep_scale=1000.0,
                    t5_feat_dim=4096,
                    t5_token_length=256,
                    rope_theta=10000.0,
                )
        with t("dit_load_checkpoint"):
            # FSDP syncs weights
            if local_rank == 0:
                if not os.environ.get("NO_LOAD", False):
                    model.load_state_dict(load_file(dit_checkpoint_path))

        with t("fsdp_dit"):
            # self.dit = (
            #     setup_fsdp_sync(
            #         model,
            #         device_id=device_id,
            #         param_dtype=torch.bfloat16,
            #         auto_wrap_policy=partial(
            #             lambda_auto_wrap_policy,
            #             lambda_fn=lambda m: m in model.blocks,
            #         ),
            #     )
            #     if world_size > 1
            #     else model.to(self.device)
            # )
            self.dit = model
            self.dit.to(self.device)
            self.dit.eval()
            if os.environ.get("COMPILE_DIT") == "1":
                print("COMPILING DIT ...")
                model = torch.compile(model)
            else:
                print("NOT COMPILING DIT ...")

        vae_stats = json.load(open(vae_stats_path))
        self.vae_mean = torch.Tensor(vae_stats["mean"]).to(self.device)
        self.vae_std = torch.Tensor(vae_stats["std"]).to(self.device)

        t.print_stats()

    def get_conditioning(self, prompts, *, zero_last_n_prompts: int):
        B = len(prompts)
        assert (
            0 <= zero_last_n_prompts <= B
        ), f"zero_last_n_prompts should be between 0 and {B}, got {zero_last_n_prompts}"
        tokenize_kwargs = dict(
            prompt=prompts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        t5_toks = self.t5_tokenizer(**tokenize_kwargs, max_length=MAX_T5_TOKEN_LENGTH)
        caption_input_ids_t5 = t5_toks["input_ids"]
        caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
        del t5_toks

        assert caption_input_ids_t5.shape == (B, MAX_T5_TOKEN_LENGTH)
        assert caption_attention_mask_t5.shape == (B, MAX_T5_TOKEN_LENGTH)

        if zero_last_n_prompts > 0:
            # Zero the last N prompts
            caption_input_ids_t5[-zero_last_n_prompts:] = 0
            caption_attention_mask_t5[-zero_last_n_prompts:] = False

        caption_input_ids_t5 = caption_input_ids_t5.to(self.device, non_blocking=True)
        caption_attention_mask_t5 = caption_attention_mask_t5.to(
            self.device, non_blocking=True
        )

        y_mask = [caption_attention_mask_t5]
        y_feat = []

        y_feat.append(
            self.t5_enc(
                caption_input_ids_t5, caption_attention_mask_t5
            ).last_hidden_state.detach()
        )
        # Sometimes returns a tensor, othertimes a tuple, not sure why
        # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
        assert tuple(y_feat[-1].shape) == (B, MAX_T5_TOKEN_LENGTH, 4096)
        assert y_feat[-1].dtype == torch.float32

        return dict(y_mask=y_mask, y_feat=y_feat)

    def get_packed_indices(self, y_mask, *, lT, lW, lH):
        patch_size = 2
        N = lT * lH * lW // (patch_size**2)
        assert len(y_mask) == 1
        packed_indices = compute_packed_indices(N, y_mask)
        self.move_to_device_(packed_indices)
        return packed_indices

    def move_to_device_(self, sample):
        if isinstance(sample, dict):
            for key in sample.keys():
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].to(self.device, non_blocking=True)

    @torch.inference_mode(mode=True)
    def run(self, args, stream_results):
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

        generator = torch.Generator(device=self.device)
        generator.manual_seed(args["seed"])

        assert (
            len(args["prompt"]) == 1
        ), f"Expected exactly one prompt, got {len(args['prompt'])}"
        prompt = args["prompt"][0]
        neg_prompt = args["negative_prompt"][0] if len(args["negative_prompt"]) else ""
        B = 1

        w = args["width"]
        h = args["height"]
        t = args["num_frames"]
        batch_cfg = args["mochi_args"]["batch_cfg"]
        sample_steps = args["mochi_args"]["num_inference_steps"]
        cfg_schedule = args["mochi_args"].get("cfg_schedule")
        assert (
            len(cfg_schedule) == sample_steps
        ), f"cfg_schedule must have length {sample_steps}, got {len(cfg_schedule)}"
        sigma_schedule = args["mochi_args"].get("sigma_schedule")
        if sigma_schedule:
            assert (
                len(sigma_schedule) == sample_steps + 1
            ), f"sigma_schedule must have length {sample_steps + 1}, got {len(sigma_schedule)}"
        assert (t - 1) % 6 == 0, f"t - 1 must be divisible by 6, got {t - 1}"

        if batch_cfg:
            sample_batched = self.get_conditioning(
                [prompt] + [neg_prompt], zero_last_n_prompts=B if neg_prompt == "" else 0
            )
        else:
            sample = self.get_conditioning([prompt], zero_last_n_prompts=0)
            sample_null = self.get_conditioning([neg_prompt] * B, zero_last_n_prompts=B if neg_prompt == "" else 0)

        spatial_downsample = 8
        temporal_downsample = 6
        latent_t = (t - 1) // temporal_downsample + 1
        latent_w, latent_h = w // spatial_downsample, h // spatial_downsample

        latent_dims = dict(lT=latent_t, lW=latent_w, lH=latent_h)
        in_channels = 12
        z = torch.randn(
            (B, in_channels, latent_t, latent_h, latent_w),
            device=self.device,
            generator=generator,
            dtype=torch.float32,
        )

        if batch_cfg:
            sample_batched["packed_indices"] = self.get_packed_indices(
                sample_batched["y_mask"], **latent_dims
            )
            z = repeat(z, "b ... -> (repeat b) ...", repeat=2)
        else:
            sample["packed_indices"] = self.get_packed_indices(
                sample["y_mask"], **latent_dims
            )
            sample_null["packed_indices"] = self.get_packed_indices(
                sample_null["y_mask"], **latent_dims
            )

        def model_fn(*, z, sigma, cfg_scale):
            if batch_cfg:
                with torch.autocast(device_type='cuda' if z.is_cuda else 'cpu', dtype=torch.bfloat16 if z.is_cuda else torch.float32):
                    out = self.dit(z, sigma, **sample_batched)
                out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
            else:
                nonlocal sample, sample_null
                with torch.autocast(device_type='cuda' if z.is_cuda else 'cpu', dtype=torch.bfloat16 if z.is_cuda else torch.float32):
                    out_cond = self.dit(z, sigma, **sample)
                    out_uncond = self.dit(z, sigma, **sample_null)
            assert out_cond.shape == out_uncond.shape
            return out_uncond + cfg_scale * (out_cond - out_uncond), out_cond

        for i in range(0, sample_steps):
            if not sigma_schedule:
                raise NotImplementedError("sigma_schedule is required. Specifying shift has not been wired up to the CLI or UI.")
                sigma = 1 - i / sample_steps
                sigma_next = 1 - (i + 1) / sample_steps

                sigma = shift_sigma(sigma, shift=shift)
                sigma_next = shift_sigma(sigma_next, shift=shift)
                dsigma = sigma - sigma_next
            else:
                sigma = sigma_schedule[i]
                dsigma = sigma - sigma_schedule[i + 1]

            # `pred` estimates `z_0 - eps`.
            pred, output_cond = model_fn(
                z=z,
                sigma=torch.full(
                    [B] if not batch_cfg else [B * 2], sigma, device=z.device
                ),
                cfg_scale=cfg_schedule[i],
            )
            pred = pred.to(z)
            output_cond = output_cond.to(z)

            if stream_results:
                yield i / sample_steps, None, False
            z = z + dsigma * pred

        cp_rank, cp_size = cp.get_cp_rank_size()
        if batch_cfg:
            z = z[:B]
        z = z.tensor_split(cp_size, dim=2)[cp_rank]  # split along temporal dim
        samples = unnormalize_latents(z.float(), self.vae_mean, self.vae_std)
        with torch.amp.autocast(device_type='cuda' if samples.is_cuda else 'cpu', dtype=torch.bfloat16 if samples.is_cuda else torch.float32):
            samples = self.decoder(samples)

        samples = cp_conv.gather_all_frames(samples)

        samples = samples.float()
        samples = (samples + 1.0) / 2.0
        samples.clamp_(0.0, 1.0)

        frames = rearrange(samples, "b c t h w -> t b h w c").cpu().numpy()

        if stream_results:
            yield 1.0, frames, True

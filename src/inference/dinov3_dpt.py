"""Inference for the released DINOv3 + DPT CoralscapesV2 checkpoints.

The EPFL-ECEO ``coralscapesv2-dinov3-*-lora-dpt*`` repos are not ``transformers``
AutoModels. Each ships ``coralscapes_hub_model.py`` describing a frozen DINOv3
ViT encoder (LoRA adapters already merged) plus a DPT decoder, and all weights
live in a single ``model.safetensors``. Loading therefore needs a dynamic import
of that module rather than ``AutoModel``.

The published loader instantiates the encoder with
``AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")``, a gated
repo, purely to get the architecture before overwriting every weight from the
Apache-2.0 ``model.safetensors``. That download is unnecessary: the encoder
hyperparameters are recoverable from the checkpoint's own tensor shapes, so the
architecture is rebuilt with ``AutoModel.from_config`` and no gated artefact is
contacted.

Model output ID 0 is void; IDs 1..95 are the native CoralscapesV2 classes, which
are folded into the project taxonomy by name, exactly like the SegFormer-95
checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from src.data.v2_label_map import V2_CLASSES, V2_TO_ID, v2_name_to_class

MODEL_TYPE = "dinov3_dpt_segmenter"
DEFAULT_REPO = "EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class"
# Sink for the model's void label and for fine labels with no project destination.
PROJECT_SINK = len(V2_CLASSES)
VOID = 255


def is_dinov3_dpt(model_path: str | Path) -> bool:
    """True when the path is a DINOv3+DPT checkpoint rather than a SegFormer one."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == MODEL_TYPE


def _load_hub_module(model_dir: Path) -> ModuleType:
    module_path = model_dir / "coralscapes_hub_model.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"{module_path} is missing; a DINOv3+DPT checkpoint must ship its hub model code"
        )
    spec = importlib.util.spec_from_file_location("coralscapes_hub_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import hub model code from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER_PREFIX = "encoder."
HEAD_DIM = 64  # DINOv3 keeps head_dim fixed at 64 across ViT-S/B/L


def infer_encoder_config(weights_path: str | Path):
    """Rebuild the DINOv3 ViT configuration from the checkpoint's tensor shapes.

    The gated encoder repo is only needed for its ``config.json``; every
    hyperparameter that affects the architecture is visible in the weights, and
    the remaining defaults (``hidden_act="gelu"``, ``rope_theta=100``,
    ``layer_norm_eps=1e-5``) are the DINOv3 ones. Loading the result with
    ``strict=True`` fails loudly if the reconstruction is wrong.
    """
    from safetensors import safe_open
    from transformers import DINOv3ViTConfig

    with safe_open(str(weights_path), framework="pt") as handle:
        keys = [key for key in handle.keys() if key.startswith(ENCODER_PREFIX)]
        if not keys:
            raise RuntimeError(f"{weights_path} has no '{ENCODER_PREFIX}' tensors")
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}

    def shape(suffix: str) -> tuple[int, ...]:
        key = ENCODER_PREFIX + suffix
        if key not in shapes:
            raise RuntimeError(f"Cannot infer encoder config: missing {key}")
        return shapes[key]

    hidden_size = shape("embeddings.cls_token")[-1]
    patch_size = shape("embeddings.patch_embeddings.weight")[-1]
    num_layers = len({key.split(".")[3] for key in keys if key.startswith(f"{ENCODER_PREFIX}model.layer.")})
    use_gated_mlp = any("mlp.gate_proj" in key for key in keys)
    intermediate_size = shape("model.layer.0.mlp.down_proj.weight")[-1]
    layer = "model.layer.0.attention."
    return DINOv3ViTConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden_size // HEAD_DIM,
        patch_size=patch_size,
        num_register_tokens=shape("embeddings.register_tokens")[1],
        use_gated_mlp=use_gated_mlp,
        query_bias=f"{ENCODER_PREFIX}{layer}q_proj.bias" in shapes,
        key_bias=f"{ENCODER_PREFIX}{layer}k_proj.bias" in shapes,
        value_bias=f"{ENCODER_PREFIX}{layer}v_proj.bias" in shapes,
        proj_bias=f"{ENCODER_PREFIX}{layer}o_proj.bias" in shapes,
        mlp_bias=f"{ENCODER_PREFIX}model.layer.0.mlp.down_proj.bias" in shapes,
        # The encoder is a frozen feature extractor, not a backbone.
        apply_layernorm=False,
        reshape_hidden_states=False,
    )


def load_dinov3_dpt(
    model_path: str | Path,
    device: str = "cpu",
    local_files_only: bool = False,
):
    """Load a DINOv3+DPT segmenter and its image processor.

    ``model_path`` must be a local checkpoint directory: this repo keeps
    pretrained weights under ``models/pretrained`` so runs are reproducible
    without network access. The encoder architecture is rebuilt from the
    checkpoint itself, so the gated Facebook repo is never required.
    """
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"DINOv3+DPT checkpoint directory not found: {model_dir}")
    module = _load_hub_module(model_dir)
    encoder_config = infer_encoder_config(model_dir / "model.safetensors")

    # The third-party loader builds the encoder via a gated repo before
    # overwriting every weight from the local checkpoint. Substitute it with the
    # locally reconstructed architecture for the duration of the load only.
    from transformers import AutoModel

    original = AutoModel.__dict__["from_pretrained"]

    def _from_local_config(pretrained_model_name_or_path=None, *args, **kwargs):
        kwargs.pop("local_files_only", None)
        kwargs.pop("cache_dir", None)
        kwargs.pop("trust_remote_code", None)
        return AutoModel.from_config(encoder_config, **kwargs)

    AutoModel.from_pretrained = staticmethod(_from_local_config)
    try:
        model = module.Dinov3DPTSegmenter.from_pretrained(
            model_dir, map_location=device, local_files_only=local_files_only
        )
    finally:
        AutoModel.from_pretrained = original
    model.eval()
    return model, model.processor


LABELS_FILENAME = "id2label.json"
# Canonical 95-class names, versioned so a freshly downloaded checkpoint is
# self-describing: the DINOv3+DPT repos ship ``num_classes`` but no label names.
REPO_LABELS_PATH = Path(__file__).resolve().parents[2] / "configs" / "dinov3_id2label.json"


def load_id2label(model_path: str | Path) -> dict[int, str]:
    """Read the CoralscapesV2 label names for a DINOv3+DPT checkpoint.

    The checkpoint-local ``id2label.json`` (installed by
    ``scripts/fetch_pretrained.py``) takes precedence; otherwise the canonical
    copy in ``configs/dinov3_id2label.json`` is used.
    """
    path = Path(model_path) / LABELS_FILENAME
    if not path.is_file():
        path = REPO_LABELS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The DINOv3+DPT checkpoint needs the CoralscapesV2 "
            f"95-class id2label; restore configs/dinov3_id2label.json."
        )
    raw = json.loads(path.read_text())
    return {int(index): str(name) for index, name in raw.items()}


def dinov3_project_ids(labels_by_id: dict[int, str], num_model_classes: int) -> list[int]:
    """Map each model output ID to a project class (``PROJECT_SINK`` = discard).

    The DINOv3 config does not carry the label names, so callers pass the
    CoralscapesV2 id2label the checkpoint was trained on.
    """
    ids: list[int] = []
    for index in range(num_model_classes):
        name = str(labels_by_id.get(index, "")).strip()
        # ID 0 is the model's explicit void category.
        if index == 0 or not name or name.lower() in {"void", "unlabeled"}:
            ids.append(PROJECT_SINK)
            continue
        project_class = v2_name_to_class(name)
        ids.append(V2_TO_ID[project_class] if project_class else PROJECT_SINK)
    return ids


def predict_project_map(
    model,
    processor,
    image,
    project_ids: list[int],
    device: str = "cpu",
    size: tuple[int, int] = (1376, 768),
) -> np.ndarray:
    """Single-pass inference folded into project classes.

    ``image`` is a PIL RGB image. Returns a ``uint8`` map at ``size`` (W, H) with
    project class IDs, and 255 where the model predicted void.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage

    width, height = size
    if image.size != (width, height):
        image = image.resize((width, height), PILImage.Resampling.BILINEAR)
    pixel_values = processor(images=image, return_tensors="pt", do_resize=False)["pixel_values"]
    pixel_values = pixel_values.to(device)
    num_project = len(V2_CLASSES)
    with torch.inference_mode():
        logits = model(pixel_values).float()
        if logits.shape[1] != len(project_ids):
            raise RuntimeError(
                f"Checkpoint produced {logits.shape[1]} classes but {len(project_ids)} IDs were mapped"
            )
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        probabilities = logits.softmax(dim=1)
        grouped = torch.zeros(
            (1, num_project + 1, height, width), dtype=probabilities.dtype, device=probabilities.device
        )
        group_ids = torch.as_tensor(project_ids, dtype=torch.long, device=probabilities.device)
        grouped.scatter_add_(
            1, group_ids.view(1, -1, 1, 1).expand(1, -1, height, width), probabilities
        )
        project_map = grouped.argmax(dim=1)[0].to("cpu", dtype=torch.uint8).numpy()
    project_map[project_map == num_project] = VOID
    return project_map

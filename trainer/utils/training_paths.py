from pathlib import Path
import os
import trainer.constants as train_cst
from trainer.utils.style_detection import detect_styles_in_prompts
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import ImageModelType

def get_checkpoints_output_path(task_id: str, repo_name: str) -> str:
    return str(Path(train_cst.OUTPUT_CHECKPOINTS_PATH) / task_id / repo_name)

def get_image_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    base_path = str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)
    if os.path.isdir(base_path):
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(base_path, files[0])
    return base_path

def get_image_training_images_dir(task_id: str) -> str:
    return str(Path(train_cst.IMAGE_CONTAINER_IMAGES_PATH) / task_id / "img")

def _detect_style_from_captions(train_data_dir: str) -> bool:
    """Deteksi style dari caption .txt. Return False jika ragu."""
    candidates = [
        os.path.join(train_data_dir, "5_lora style"),
        train_data_dir,
    ]
    prompts = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for file in os.listdir(d):
            if file.endswith(".txt"):
                try:
                    with open(os.path.join(d, file), "r") as f:
                        prompts.append(f.read().strip())
                except Exception:
                    pass
        if prompts:
            break
    if not prompts:
        return False
    styles = detect_styles_in_prompts(prompts)
    return bool(styles)

def get_image_training_config_template_path(model_type: str, train_data_dir: str, trigger_word: str | None = None) -> tuple[str, bool]:
    model_type = model_type.lower()
    if model_type == ImageModelType.SDXL.value:
        # [dethrone] style = trigger_word None (ground-truth validator: style task = trigger NULL).
        # Deterministik; ganti deteksi caption (cuma proxy). ds_prefix di-strip saat training,
        # tapi trigger_word ke-pass via --trigger-word -> sinyal valid di runtime.
        is_style = trigger_word is None or not str(trigger_word).strip()
        tmpl = "base_diffusion_sdxl_style.toml" if is_style else "base_diffusion_sdxl_person.toml"
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / tmpl), is_style

    elif model_type == ImageModelType.FLUX.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_flux.toml"), False
    elif model_type in (ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value):
        is_style = _detect_style_from_captions(train_data_dir)
        template = ("base_diffusion_zimage.yaml"
                    if model_type == ImageModelType.Z_IMAGE.value
                    else "base_diffusion_qwen_image.yaml")
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / template), is_style

def get_image_training_zip_save_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_tourn.zip")

def get_text_dataset_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_train_data.json")

def get_axolotl_dataset_paths(dataset_filename: str) -> tuple[str, str]:
    data_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["data"]) / dataset_filename)
    root_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["root"]) / dataset_filename)
    return data_path, root_path

def get_axolotl_base_config_path(dataset_type) -> str:
    root_dir = Path(train_cst.AXOLOTL_DIRECTORIES["root"])
    if isinstance(dataset_type, (InstructTextDatasetType, DpoDatasetType)):
        return str(root_dir / "base.yml")
    elif isinstance(dataset_type, GrpoDatasetType):
        return str(root_dir / "base_grpo.yml")
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")

def get_text_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    return str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)

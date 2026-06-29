#!/usr/bin/env python3

"""
everything u are 10
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import re
import threading
import time
import yaml
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.category_detector import detect_category, read_caption_texts  # [F2] scope recipe Qwen person
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType
from auto_caption import auto_caption_dataset


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None

def compute_aitoolkit_steps(dataset_size: int,
                            per_image: int = 200,    # [dethrone] 100->200
                            min_steps: int = 2000,   # [dethrone] 600->2000: formula lama hasilin ~1400 step (=angka KALAH challenger). Floor 2000 = atas angka menang bos. Qwen->2000-3000, Z->2000.
                            max_steps: int = 2000) -> int:
    """Step proporsional ke jumlah gambar (anti-overfit untuk dataset kecil).
    [dethrone] Pelajaran final 11-Jun: pada Qwen/Z, step LEBIH BANYAK menang
    (bos 2000-2500 ngalahin challenger 1440). max_steps di-cap di caller (3000 Qwen / 2000 Z)."""
    if dataset_size <= 0:
        return min_steps
    steps = dataset_size * per_image
    return max(min_steps, min(steps, max_steps))

def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count



def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


# ============ [dethrone] routing per-kategori SDXL (HYBRID, Final Round Intel) ============
# Kategori BUKAN ditebak liar: style = trigger_word None (ground-truth). logo/design = keyword
# HIGH-PRECISION (fraction-based, bias ke default). person/product/social + ambigu = DEFAULT plain-64.
# Asimetri downside: DoRA-default = blunder product -26% (bukti T3); plain-default = cuma suboptimal.
CATEGORY_FRACTION_THRESHOLD = 0.6   # >=60% caption harus match -> task BENERAN logo/design, bukan sebutan insidental
# logo: istilah logo-DESIGN (bukan "logo" telanjang yg bisa muncul sbg fitur produk, mis. "debossed 'AG' logo")
LOGO_PATTERNS = [r'\bvector\b', r'brand\s*mark', r'\bwordmark\b', r'\bmonogram\b', r'\blogotype\b',
                 r'logo\s+(mark|design|concept|collection|artwork|icon|grid)',
                 r'(minimalist|geometric|flat|abstract|modern|line-art)\s+logo']
DESIGN_PATTERNS = [r'\bUI\b', r'\bUX\b', r'user interface', r'\bwireframe\b', r'\bdashboard\b',
                   r'app\s+(screen|interface|ui)', r'mobile\s+app', r'\bwebsite\b', r'landing\s+page',
                   r'\bscreenshot\b', r'\bSaaS\b', r'navigation\s+bar', r'sign[- ]?up\s+screen']
SDXL_NETWORK_BY_CATEGORY = {
    "style":   {"network_module": "lycoris.kohya", "network_dim": 32, "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "algo=lora", "dora_wd=True", "loraplus_lr_ratio=16", "dropout=0"]},
    "logo":    {"network_module": "lycoris.kohya", "network_dim": 32, "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "algo=lora", "dora_wd=True", "dropout=0"]},
    "design":  {"network_module": "lycoris.kohya", "network_dim": 32, "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "algo=lora", "dora_wd=True", "dropout=0"]},
    # DEFAULT (person/product/social/ambigu): plain LoRA-64, NO DoRA, NO conv. (person DoRA-64 = hipotesis sweep, bukan tahap-1)
    "default": {"network_module": "networks.lora", "network_dim": 64, "network_alpha": 64, "network_args": []},
}


def _frac_match(prompts, patterns):
    if not prompts:
        return 0.0
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    return sum(1 for p in prompts if rx.search(p)) / len(prompts)


def detect_image_category(trigger_word, prompts):
    """style|logo|design|default. High-precision: ragu -> default plain-64."""
    if trigger_word is None or not str(trigger_word).strip():
        return "style"
    logo = _frac_match(prompts, LOGO_PATTERNS)
    design = _frac_match(prompts, DESIGN_PATTERNS)
    if logo >= CATEGORY_FRACTION_THRESHOLD and logo >= design:
        return "logo"
    if design >= CATEGORY_FRACTION_THRESHOLD:
        return "design"
    return "default"


# ============ [dethrone] detektor PERSON high-precision (cd0.10 utk person-SDXL) ============
# Dipakai HANYA buat pilih caption_dropout di route "default" (network tetap plain-64 utk person & product).
# Temuan FASE 2A: cd10 > cd05 di person-SDXL (no_text -0.0045, no downside). TAPI person & product
# share route "default" -> product (fortress seri, cd10 UNTESTED) GAK BOLEH kena cd10.
# Asimetri: false-negative (person->cd05) AMAN (cuma suboptimal); false-positive (product->cd10) DILARANG.
# Aturan: cd10 HANYA kalau MAYORITAS caption sinyal-person KUAT (>=60%) DAN sinyal-product nyaris nol (<20%).
PERSON_PATTERNS = [
    r'\bportrait\b', r'\bheadshot\b', r'\bselfie\b', r'\bface\b', r'\bfacial\b',
    r'\b(man|woman|person|people|guy|girl|boy|lady|male|female|gentleman)\b',
    r'\bwearing\b', r'\bsmil(e|ing)\b', r'\bbeaming\b', r'\bgrin(ning)?\b',
    r'\b(his|her)\s+(face|smile|hair|expression|eyes|head|shoulders)\b',
    r'\b(he|she)\s+(is|appears|wears|has|stands|sits|looks|holds)\b',
    r'\b(lifestyle|portrait|headshot)\s+photograph', r'\bexpression\b',
]
# sinyal PRODUCT/objek -> kalau muncul, JANGAN anggap person (lindungi fortress product)
PRODUCT_EXCLUDE_PATTERNS = [
    r'\bproduct\b', r'\bpackaging\b', r'\bbottle\b', r'\bjar\b', r'\bdevice\b', r'\bgadget\b',
    r'\bappliance\b', r'\b(lamp|mug|cup|watch|shoe|sneaker|bag|furniture|chair|sofa|table)\b',
    r'\bstudio\s+(shot|lighting|background|setup)\b',
    r'\bon\s+a\s+(white|plain|seamless|gradient|colou?red)\s+background\b',
    r'\bcommercial\s+product\b', r'\be-?commerce\b',
]
PERSON_FRACTION_MIN = 0.60      # >=60% caption harus sinyal-person kuat
PRODUCT_FRACTION_MAX = 0.20     # <20% caption boleh sinyal-product (di atas itu -> bukan person)


def is_person_dataset(prompts):
    """True HANYA kalau high-confidence person (bias-aman ke False). Dipakai utk cd0.10."""
    if not prompts:
        return False
    person = _frac_match(prompts, PERSON_PATTERNS)
    product = _frac_match(prompts, PRODUCT_EXCLUDE_PATTERNS)
    return person >= PERSON_FRACTION_MIN and product < PRODUCT_FRACTION_MAX
# ============ end detektor person ============


def _read_sdxl_prompts(train_data_dir):
    folder = os.path.join(
        train_data_dir,
        f"{cst.DIFFUSION_SDXL_REPEATS}_{cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT} {cst.DIFFUSION_DEFAULT_CLASS_PROMPT}",
    )
    prompts = []
    try:
        for fn in os.listdir(folder):
            if fn.endswith(".txt"):
                with open(os.path.join(folder, fn)) as fh:
                    prompts.append(fh.read().strip())
    except FileNotFoundError:
        pass
    return prompts
def _sweep_env(name, cast=str, default=None):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except Exception:
        return default


def _sweep_network_override():
    """[SWEEP] SDXL network override via env DETHRONE_SWEEP_NETWORK (JSON). None = normal."""
    raw = _sweep_env("DETHRONE_SWEEP_NETWORK")
    if not raw:
        return None
    net = json.loads(raw)
    print(f"[dethrone][SWEEP] SDXL network override -> {net}", flush=True)
    return net


def _apply_sweep_overrides_aitoolkit(config):
    """[SWEEP] override opt-in Z/Qwen via env (steps/cd/TE/linear). Kosong = produksi normal."""
    steps = _sweep_env("DETHRONE_SWEEP_STEPS", int)
    cd = _sweep_env("DETHRONE_SWEEP_CD", float)
    te = _sweep_env("DETHRONE_SWEEP_QWEN_TE")
    linear = _sweep_env("DETHRONE_SWEEP_LINEAR", int)
    if all(x is None for x in (steps, cd, te, linear)):
        return
    for process in config.get("config", {}).get("process", []):
        if isinstance(process.get("train"), dict):
            if steps is not None:
                process["train"]["steps"] = steps
            if te is not None:
                te_on = str(te).lower() == "true"
                process["train"]["train_text_encoder"] = te_on
                if te_on:
                    # ai-toolkit (SDTrainer.hook_before_train_loop): RAISE kalau cache/unload TE
                    # sementara TE di-train. Matiin caching+unload TE biar TE-on jalan.
                    process["train"]["cache_text_embeddings"] = False
                    process["train"]["unload_text_encoder"] = False
        if linear is not None and isinstance(process.get("network"), dict):
            process["network"]["linear"] = linear
            process["network"]["linear_alpha"] = linear
        if cd is not None:
            for ds in process.get("datasets", []):
                ds["caption_dropout_rate"] = cd
    print(f"[dethrone][SWEEP] ai-toolkit override: steps={steps} cd={cd} te={te} linear={linear}", flush=True)
# ============ end [dethrone] routing ============


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        
        # --- [F2/F3] kategori ai-toolkit (Qwen + Z) utk scope recipe MENANG person ---
        # Qwen person (F2): EMA-off + steps=12*img + no caption_dropout.
        # Z person  (F3-rev): steps=12*img + no caption_dropout  (NO EMA -> Z emang gak pakai EMA).
        # Pemenang 5FW2: Qwen person 108=12x9, Z person 168=12x14. Kategori lain (art/logo/social) & SDXL/Flux UTUH.
        ait_category = None
        if model_type in (ImageModelType.QWEN_IMAGE.value, ImageModelType.Z_IMAGE.value):
            _ait_caps = read_caption_texts(train_data_dir)
            ait_category = detect_category(_ait_caps, trigger_word)
            print(f"[F2/F3] {model_type} category={ait_category} "
                  f"({len(_ait_caps)} captions, trigger={trigger_word!r})", flush=True)
        qwen_person = (model_type == ImageModelType.QWEN_IMAGE.value and ait_category == "person")
        z_person = (model_type == ImageModelType.Z_IMAGE.value and ait_category == "person")
        person_12ximg = qwen_person or z_person   # dua-duanya: steps=12*img + buang caption_dropout

        # --- Jalur B: size-aware step untuk ai-toolkit (Z-Image/Qwen) ---
        ait_dataset_size = 0
        if os.path.exists(train_data_dir):
            ait_dataset_size = count_images_in_directory(train_data_dir)
        if ait_dataset_size > 0 and 'config' in config and 'process' in config['config']:
            if person_12ximg:
                # recipe MENANG person: steps = 12 * jumlah_gambar (Qwen 108=12x9, Z 168=12x14).
                # NO floor/clamp (sementara; uji dataset besar nanti pas re-validate). BUKAN size-aware.
                target_steps = 12 * ait_dataset_size
            else:
                ait_max = 3000 if model_type == ImageModelType.QWEN_IMAGE.value else 2000
                target_steps = compute_aitoolkit_steps(ait_dataset_size, max_steps=ait_max)
            for process in config['config']['process']:
                if 'train' in process and isinstance(process['train'], dict):
                    process['train']['steps'] = target_steps
                if 'save' in process and isinstance(process['save'], dict):
                    se = process['save'].get('save_every', 250)
                    if se > target_steps:
                        process['save']['save_every'] = max(target_steps // 4, 1)
            _tag = (f"{'Qwen' if qwen_person else 'Z'} person 12*img") if person_12ximg else "B size-aware"
            print(f"[{_tag}] ai-toolkit: {ait_dataset_size} imgs -> {target_steps} steps", flush=True)
        # --- akhir Jalur B ---

        # --- [F2/F3] buang caption_dropout utk person (Qwen + Z); EMA-off HANYA Qwen ---
        if person_12ximg and 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                tr = process.get('train')
                if isinstance(tr, dict) and qwen_person:
                    # EMA-off CUMA Qwen (pemenang 0.06825). Z pemenang NO-EMA bawaan -> JANGAN set apa2.
                    ema = tr.get('ema_config')
                    if isinstance(ema, dict):
                        ema['use_ema'] = False
                    else:
                        tr['ema_config'] = {'use_ema': False}
                for ds in (process.get('datasets') or []):
                    ds.pop('caption_dropout_rate', None)  # pemenang person TANPA cd (template paksa 0.05 -> skip)
            if qwen_person:
                print("[F2] Qwen person: EMA use_ema=false + caption_dropout dibuang (recipe menang 0.06825)", flush=True)
            else:
                print("[F3] Z person: caption_dropout dibuang (recipe menang 5FW2; EMA tetap none, network utuh)", flush=True)
        # --- akhir [F2/F3] ---

        _apply_sweep_overrides_aitoolkit(config)   # [dethrone][SWEEP] opt-in env override (Z/Qwen)

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        dataset_size = 0
        if os.path.exists(train_data_dir):
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size > 0:
                print(f"Counted {dataset_size} images in training directory", flush=True)

        size_config_loaded = False
        lrs_config = load_lrs_config(model_type, is_style)

        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                if model_type == "flux":
                    print(f"Applying model-specific config for Flux model", flush=True)
                    for key, value in lrs_settings.items():
                        config[key] = value
                else:
                    size_key = None
                    if 1 <= dataset_size <= 10:
                        size_key = "xs"
                    elif 11 <= dataset_size <= 20:
                        size_key = "s"
                    elif 21 <= dataset_size <= 30:
                        size_key = "m"
                    elif 31 <= dataset_size <= 50:
                        size_key = "l"
                    elif 51 <= dataset_size <= 1000:
                        size_key = "xl"
                    
                    if size_key and size_key in lrs_settings:
                        print(f"Applying model-specific config for size '{size_key}'", flush=True)
                        for key, value in lrs_settings[size_key].items():
                            config[key] = value
                        size_config_loaded = True
                    else:
                        print(f"Warning: No size configuration '{size_key}' found for model '{model_name}'.", flush=True)
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 64,                       # [dethrone] 32->64: rank level juara (bos product plain-LoRA rank64 menang)
                "network_alpha": 64,
                "network_args": []
            },
            235: {
                "network_dim": 64,                       # [dethrone] 32->64
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            456: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": []
            },
            467: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            699: {
                "network_dim": 96,
                "network_alpha": 96,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        _cd_default = 0.05   # [dethrone] default global SDXL cd (flux/non-person: 0.05)
        if model_type == "sdxl":
            # [dethrone] routing per-kategori (gantikan seleksi per-model lama).
            sdxl_prompts = _read_sdxl_prompts(train_data_dir)
            category = detect_image_category(trigger_word, sdxl_prompts)
            # [dethrone] person-SDXL -> cd0.10 (FASE 2A: cd10>cd05). HANYA route 'default' + high-confidence
            # person; product (juga 'default') TETAP cd05 (fortress, cd10 untested). Network tetap plain-64.
            if category == "default" and is_person_dataset(sdxl_prompts):
                _cd_default = 0.10
                print(f"[dethrone] SDXL person high-confidence -> caption_dropout 0.10", flush=True)
            net = _sweep_network_override() or SDXL_NETWORK_BY_CATEGORY[category]   # [SWEEP] env opt-in
            # defensive: kalau DoRA dipilih tapi lycoris nggak keinstall -> fallback plain-64 (jangan crash)
            if net["network_module"] == "lycoris.kohya" and importlib.util.find_spec("lycoris") is None:
                print(f"[dethrone][WARN] lycoris TIDAK terinstall -> kategori '{category}' FALLBACK ke plain-LoRA-64. "
                      f"DoRA nggak aktif! install lycoris_lora di image biar recipe jalan.", flush=True)
                net = SDXL_NETWORK_BY_CATEGORY["default"]
            config["network_module"] = net["network_module"]
            config["network_dim"] = net["network_dim"]
            config["network_alpha"] = net["network_alpha"]
            config["network_args"] = list(net["network_args"])
            print(f"[dethrone] SDXL category={category} -> module={net['network_module']} "
                  f"dim={net['network_dim']} args={net['network_args']}", flush=True)


        # Old size config search removed as requested
        if dataset_size > 0 and not size_config_loaded:
             print(f"Warning: No size-specific configuration (xs/s/m/l/xl) found for model '{model_name}' with {dataset_size} images. Using model defaults.", flush=True)
        
        # [dethrone] cd default: 0.05 global; 0.10 utk person-SDXL high-confidence (lihat is_person_dataset).
        # 0.1->0.05 match recipe juara; person cd10 dari FASE 2A. Sweep env override tetap prioritas.
        config["caption_dropout_rate"] = _sweep_env("DETHRONE_SWEEP_CD", float, _cd_default)   # [SWEEP] env opt-in
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-script/{model_type}_train_network.py",
                "--config_file", config_path
            ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    train_data_dir = train_paths.get_image_training_images_dir(args.task_id)
    _, is_style_dataset = train_paths.get_image_training_config_template_path(args.model_type, train_data_dir)
    cat_str = "style" if is_style_dataset else "person"
    auto_caption_dataset(train_data_dir, cat_str)

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())

#fuck you copiers

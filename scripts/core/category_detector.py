"""
[F2] Detektor kategori dataset Qwen — caption-based, CASCADE + fail-safe default = 'art'.

Tujuan: scope recipe MENANG Qwen person (EMA-off + steps=12*img) HANYA ke person,
tanpa ngerusak art (EMA-on, step lama), Z, atau SDXL. Dipakai oleh create_config
(scripts/image_trainer.py) via `from core.category_detector import ...`.

Pure functions (cuma `os` + `re`, no dependency berat) -> bisa di-unit-test di CPU
tanpa import pipeline penuh.

Sumber pola (caption terverifikasi turnamen 11 & 18 Jun, lihat DIFF_qwen_person.md / riset Wen):
- LOGO  : >=60% caption mengandung kata "logo" (kebukti 100% di 4 task logo lintas turnamen;
          non-logo ~0%, product 1 false-pos / 32 = 3% < threshold).
- SOCIAL: >=60% caption mengandung "headline"/"body"/"layout"/"cta" (11/11 di task social).
- PERSON: ada trigger_word DAN >=60% caption mengandung sinyal manusia
          (wearing/smiling/portrait/headshot/his/her/man/woman/suit/shirt/face).
          Prefix nama orang BEDA lintas turnamen -> JANGAN match frasa persis, pakai sinyal manusia.
- ART   : no trigger / style dijahit di prosa ("in a X style", "rendered in X").
          Juga = FAIL-SAFE default kalau ragu (recipe art aman: EMA-on, step besar).

CATATAN MATCHING (deviasi sadar dari pseudocode awal `k in c` substring):
matching dilakukan per-KATA (word-level) setelah normalize, BUKAN substring. Alasan:
substring "his"/"her"/"man" bakal false-positive ("history","where","manager"). Word-level
lebih robust dan sesuai maksud "caption ada KATA X". (lihat test trap di tests/).
"""

import os
import re


# kata kunci per kategori (single-token; dicek word-level setelah normalize)
LOGO_KEYWORDS = ("logo",)
SOCIAL_KEYWORDS = ("headline", "body", "layout", "cta")
HUMAN_KEYWORDS = (
    "wearing", "smiling", "portrait", "headshot", "his", "her",
    "man", "woman", "suit", "shirt", "face",
)
FRACTION_THRESHOLD = 0.6


def normalize_caption(text: str) -> str:
    """lowercase + tanda baca -> spasi + collapse whitespace. Konservatif (jangan gabung kata)."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)   # tanda baca jadi spasi (bukan dihapus) biar kata gak nyatu
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fraction_with_any(norm_captions, keywords) -> float:
    """Proporsi caption (0..1) yang punya >=1 keyword, dicek per-kata."""
    if not norm_captions:
        return 0.0
    hits = 0
    for c in norm_captions:
        tokens = set(c.split())
        if any(k in tokens for k in keywords):
            hits += 1
    return hits / len(norm_captions)


def detect_category(captions, trigger_word):
    """
    (captions, trigger_word) -> 'logo' | 'social' | 'person' | 'art'.

    CASCADE berurutan, berhenti di match pertama (yang paling bahaya-jika-kelewat duluan):
      1) logo   (>=60% kata 'logo')
      2) social (>=60% headline/body/layout/cta)
      3) person (trigger_word ADA  DAN  >=60% sinyal manusia)
      4) art    (fail-safe default)
    logo/social dicek SEBELUM person supaya logo/social yang kebetulan punya trigger
    tidak ke-treat sebagai person.
    """
    norm = [normalize_caption(c) for c in captions if c and c.strip()]
    if not norm:
        return "art"   # no caption -> fail-safe art

    if _fraction_with_any(norm, LOGO_KEYWORDS) >= FRACTION_THRESHOLD:
        return "logo"
    if _fraction_with_any(norm, SOCIAL_KEYWORDS) >= FRACTION_THRESHOLD:
        return "social"
    if (trigger_word and str(trigger_word).strip()
            and _fraction_with_any(norm, HUMAN_KEYWORDS) >= FRACTION_THRESHOLD):
        return "person"
    return "art"


def read_caption_texts(train_data_dir):
    """Baca SEMUA caption .txt (rekursif) di bawah train_data_dir. Return [] kalau gak ada.

    Rekursif karena layout ai-toolkit = train_data_dir/{repeat}_lora style/*.txt
    (repeat=1 utk Qwen/Z, 5 utk SDXL) — rekursif aman utk semua layout.
    """
    captions = []
    if not train_data_dir or not os.path.isdir(train_data_dir):
        return captions
    for root, _dirs, files in os.walk(train_data_dir):
        for fn in files:
            if fn.endswith(".txt"):
                try:
                    with open(os.path.join(root, fn), "r", encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read().strip()
                    if txt:
                        captions.append(txt)
                except Exception:
                    pass
    return captions

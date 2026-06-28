"""
[F2] Unit test detektor kategori Qwen (scripts/core/category_detector.py).
Self-contained: caption sintetik yang meniru pola dataset turnamen 11 & 18 Jun
(person David/Evelyn, logo Brandmark, social AuraVerse, art Dreamlike/Cubism).
Jalan tanpa GPU/dataset: `python3 scripts/core/tests/test_category_detector.py`

Smoke-test pada caption ASLI (dataset 11/18 Jun) sudah dijalankan & LULUS 6/6 + 5 guard
(lihat TUGAS1_detector_result di laporan). Test ini menjaga regresi logika.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.category_detector import detect_category  # noqa: E402

PERSON = [  # mirip 77b5e0fb (David) — portrait/wearing/smiling/suit
    "Realistic photographic portrait of David Miller smiling gently, wearing a navy blue business suit.",
    "David Miller looking thoughtful on a city street, wearing a casual dark jacket and a light shirt.",
    "Close-up realistic photo of David Miller, slight grin, wearing a simple crew neck t-shirt.",
    "David Miller seated at a desk, wearing glasses, his face lit by soft window light.",
]
LOGO = [  # mirip 7489f54c — 'Logo for ...' + juga ada 'layout'/'body' (uji cascade)
    "Brandmark_Essentials, Logo for 'NovaTech', abstract mark. Layout: centered. Visual style: flat vector.",
    "Brandmark_Essentials, Logo for 'Wild Bloom', line-art flower. Layout: mark left of name. Body of text below.",
    "Brandmark_Essentials, Logo for 'Gourmet', laurel emblem. Layout: circular stamp. Minimalist logo design.",
]
SOCIAL = [  # mirip 39813ec7 — Headline/Body/CTA/Layout
    "AuraVerse: Instagram Story. Headline: 'Radiant Skin'. Body: 'Introducing Glow Elixir'. CTA: 'Shop Now'. Layout: thirds.",
    "AuraVerse: Facebook Post. Headline: 'Summer Savings'. Body: 'Up to 50% off'. CTA: 'Browse Deals'. Layout: diagonal.",
    "AuraVerse: Blog image. Headline: 'Unleash Your Artist'. Body: 'Read our tips'. Layout: centered block.",
]
ART_NO_HUMAN = [  # mirip f9459ba8 (Dreamlike) — style di prosa, no trigger
    "A soaring ethereal airship glowing with runes against a nebula, in a Dreamlike and Digital Art style.",
    "A lone astronaut gazing into an alien cityscape, in a Dreamlike and Digital Art style.",
    "A steampunk owl with gear-work wings on a metallic branch, in a Dreamlike and Digital Art style.",
]
ART_CUBISM = [  # mirip a871190c — ADA 'portrait'/'his' tapi minoritas, no trigger -> art
    "A fragmented cityscape at twilight, in Cubism and Acrylic Painting, bold geometric planes.",
    "A portrait of a musing musician, his guitar deconstructed, in Cubism and Acrylic Painting.",
    "An architectural interior from many viewpoints, in Cubism and Acrylic Painting, cool grays.",
    "A still life of fruit and bottles, in Cubism and Acrylic Painting, sharp angular shards.",
]
TRAP = [  # substring his/her/man di history/where/manager -> JANGAN person
    "The history of this place, where the manager works.",
    "A history book about somewhere, nothing human at all.",
    "Where does the history of management begin?",
]


def check(name, captions, trigger, expected):
    got = detect_category(captions, trigger)
    status = "OK" if got == expected else "FAIL"
    print(f"[{status}] {name:<34} trigger={str(trigger):<16} -> {got} (exp {expected})")
    return got == expected


def main():
    ok = True
    ok &= check("person David", PERSON, "David Miller", "person")
    ok &= check("logo + trigger (cascade)", LOGO, "Brandmark_Essentials", "logo")
    ok &= check("social + trigger", SOCIAL, "AuraVerse", "social")
    ok &= check("art Dreamlike (no trigger)", ART_NO_HUMAN, None, "art")
    ok &= check("art Cubism (no trigger)", ART_CUBISM, None, "art")
    # guards
    ok &= check("person TANPA trigger -> art", PERSON, None, "art")
    ok &= check("Cubism + trigger keliru -> art", ART_CUBISM, "X", "art")  # human% < 60
    ok &= check("empty -> art", [], "X", "art")
    ok &= check("trap substring -> art", TRAP, "X", "art")
    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILED ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

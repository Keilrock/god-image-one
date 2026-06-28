# F2 TUGAS 3 — Test `create_config` (CPU, no train)

> Verifikasi scope recipe Qwen person (TUGAS 2) dengan memanggil `create_config()` ASLI di CPU
> (no GPU, no train). Harness: stub `fiber`/`transformers` (gak dipake create_config), monkeypatch
> path ke lokal, dataset caption ASLI 11/18 Jun. Fungsi yang dites = `scripts/image_trainer.py`
> hasil edit TUGAS 2 (bukan replika).

## Hasil per skenario (config yang DI-GENERATE create_config)

| skenario | model_type | trigger | →detect | use_ema | steps | caption_dropout |
|---|---|---|---|---|---|---|
| **1. Qwen PERSON** (9 img) | qwen-image | David Miller | person | **false** | **108** (12×9) | **ABSENT** |
| **2. Qwen ART** (13 img, no trigger) | qwen-image | None | art | **true** | **2600** (size-aware) | **0.05** |
| **3. Qwen PERSON sim** (30 img) | qwen-image | David Miller | person | **false** | **360** (12×30) | **ABSENT** |
| **4. Z-IMAGE** (pakai data person!) | z-image | David Miller | — (bukan qwen) | ABSENT (template Z) | **2000** (size-aware) | 0.05 |
| **5a. Qwen LOGO** (trigger) | qwen-image | Brandmark_Essentials | logo | **true** | **3000** (size-aware) | **0.05** |
| **5b. Qwen SOCIAL** (trigger) | qwen-image | AuraVerse | social | **true** | **2200** (size-aware) | **0.05** |
| **6. SDXL** (branch lain) | sdxl | MyTrigger | default | (no EMA) | (toml, no qwen-step) | 0.05 |

## Penegasan: CUMA Qwen-person yang kena recipe khusus

✅ **Qwen PERSON** (skenario 1 & 3) — satu-satunya yang dapet recipe MENANG:
- `use_ema = false` (template aslinya true)
- `steps = 12 × jumlah_gambar` (108 utk 9img, 360 utk 30img — formula scale, BUKAN size-aware floor 2000)
- `caption_dropout` DIBUANG (template paksa 0.05 → di-skip)

✅ **Semua kategori Qwen LAIN + Z + SDXL = UTUH** (logika existing, gak kena fix person):
- **Qwen ART** → EMA **on**, steps size-aware (2600), cd 0.05. (art butuh EMA-on — gak rusak ✔)
- **Qwen LOGO** → else branch: EMA on, steps size-aware (3000), cd 0.05.
- **Qwen SOCIAL** → else branch: EMA on, steps size-aware (2200), cd 0.05.
- **Z-IMAGE** (sengaja dikasih data person + trigger) → **tetap TIDAK kena** karena guard `model_type==qwen-image`. steps size-aware 2000, EMA template (none), cd 0.05.
- **SDXL** → branch toml terpisah, `qwen_person` gak pernah disentuh. category=default→plain-64, cd0.05, no EMA/qwen-step. No crash.

## Assertions (semua LULUS ✅)
```
person9:  EMA off | steps==108 (12x9) | NO caption_dropout
art13:    EMA ON  | steps size-aware (2600) != 156 | cd 0.05
person30: EMA off | steps==360 (12x30) | NO caption_dropout
z-image:  steps != 108 (=2000) -> tidak kena fix person | EMA tidak dipaksa-off
Qwen LOGO:   EMA on + steps size-aware (3000) -> else branch | cd 0.05
Qwen SOCIAL: EMA on + steps size-aware (2200) -> else branch | cd 0.05
```

## Sweep inert (skenario 6 dari user)
Dijalankan **tanpa** `DETHRONE_SWEEP_*` env var → `_apply_sweep_overrides_aitoolkit` no-op → **F2-scoping yang berlaku**.
Konfirmasi: nilai person (EMA-off, steps 12×img, no-cd) muncul di config tanpa gangguan sweep.
(F2-scoping jalan SEBELUM hook sweep; di turnamen validator gak set env sweep → aman.)

## Catatan
- Z template (`base_diffusion_zimage.yaml`) emang gak punya `ema_config` → `use_ema` ABSENT itu memang kondisi asli template (bukan diubah F2).
- Harness CPU (stub fiber/transformers, monkeypatch path) ada di `/ephemeral/f2test/` (ephemeral, gak di-commit). Logika detektor dijaga oleh unit test `scripts/core/tests/test_category_detector.py`.
- Re-validate GPU (1 run/task) nanti setelah semua recipe masuk.

#!/usr/bin/env python3
"""
split_dataset.py — PENGGANTI fetch_task_dataset.py (yg gak pernah ke-commit di repo).

`fetch_task_dataset.py` yg dirujuk README_DETHRONE.md TIDAK ADA di repo/git-history, dan
dethrone_tasks.json TIDAK punya presigned URL. Jadi alih-alih pull dari URL, script ini
bikin layout data/<T>/{train,test} dari SUMBER LOKAL (zip/dir hasil scp cadangan).

Input  : folder ATAU zip berisi gambar + .txt caption (flat, atau 1 subfolder).
Output : <out>/train/ dan <out>/test/  (masing2 gambar + .txt pasangannya).
Split  : deterministik (seeded). test = fraksi (--split 0.2) ATAU jumlah tetap (--test-n 5).

Contoh (sesuai runbook, setelah scp zip per task):
  python3 split_dataset.py --src QF_raw.zip --out data/QF --test-n 1 --seed 42
  python3 split_dataset.py --src T1_raw/    --out data/T1 --test-n 5
  python3 split_dataset.py --src T5_raw.zip --out data/T5 --test-n 3
  python3 split_dataset.py --src T6_raw.zip --out data/T6 --test-n 3
  # fraksi: --split 0.2 (dipakai kalau --test-n gak diisi)

Catatan: held-out lokal != held-out validator (validate_eval pakai tol longgar; yg penting RANKING).
"""
import argparse, os, random, shutil, sys, tempfile, zipfile

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _materialize(src):
    """return dir berisi file (extract zip ke tmp kalau perlu). (dir, cleanup_or_None)"""
    if os.path.isdir(src):
        return src, None
    if zipfile.is_zipfile(src):
        tmp = tempfile.mkdtemp(prefix="ds_")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        items = [e for e in os.listdir(tmp) if not e.startswith(".")]
        if len(items) == 1 and os.path.isdir(os.path.join(tmp, items[0])):
            return os.path.join(tmp, items[0]), tmp
        return tmp, tmp
    sys.exit(f"[FATAL] --src bukan dir & bukan zip: {src}")


def _collect_pairs(d):
    """{stem: (img_path, txt_path|None)} utk semua gambar di d (rekursif)."""
    pairs = {}
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXT:
                full = os.path.join(root, f)
                txt = os.path.join(root, stem + ".txt")
                pairs[full] = (full, txt if os.path.isfile(txt) else None)
    return list(pairs.values())


def _copy_pairs(pairs, dst):
    os.makedirs(dst, exist_ok=True)
    n_txt = 0
    for img, txt in pairs:
        shutil.copy2(img, os.path.join(dst, os.path.basename(img)))
        if txt:
            shutil.copy2(txt, os.path.join(dst, os.path.basename(txt)))
            n_txt += 1
    return n_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zip atau dir berisi gambar + .txt")
    ap.add_argument("--out", required=True, help="output root -> <out>/train, <out>/test")
    ap.add_argument("--split", type=float, default=0.2, help="fraksi test (dipakai kalau --test-n kosong)")
    ap.add_argument("--test-n", type=int, default=None, help="jumlah gambar test TETAP (override --split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir, cleanup = _materialize(args.src)
    try:
        pairs = _collect_pairs(src_dir)
    finally:
        pass
    if not pairs:
        sys.exit(f"[FATAL] gak nemu gambar di {args.src}")

    n = len(pairs)
    n_test = args.test_n if args.test_n is not None else max(1, round(n * args.split))
    n_test = min(n_test, n - 1) if n > 1 else 0  # sisain minimal 1 train

    random.seed(args.seed)
    order = pairs[:]
    random.shuffle(order)
    test_pairs, train_pairs = order[:n_test], order[n_test:]

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    t_txt = _copy_pairs(train_pairs, os.path.join(args.out, "train"))
    e_txt = _copy_pairs(test_pairs, os.path.join(args.out, "test"))

    if cleanup:
        shutil.rmtree(cleanup, ignore_errors=True)

    miss = n - (sum(1 for _, t in pairs if t))
    print(f"[{args.out}] total={n}  train={len(train_pairs)} (txt {t_txt})  "
          f"test={len(test_pairs)} (txt {e_txt})  seed={args.seed}")
    if miss:
        print(f"[WARN] {miss}/{n} gambar TANPA .txt caption — eval text-guided butuh caption; cek sumber.")


if __name__ == "__main__":
    main()

"""Tests for the on-device (gateway) model pipeline in edge/.

The C engine itself is tested against the same integer reference by
test_edge_c_engine_matches_reference, which only runs where a host C++
compiler (g++) and the gateway sketch folder are available.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from edge import int8
from edge.device_preprocess import box_bounds, box_resize, to_model_input
from edge.student import build_student

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# ml/ and hardware/ sit side by side under D:\farmer (farmer-web/ml, hardware/gateway_node).
GATEWAY_DIR = Path(os.environ.get("KN_GATEWAY_DIR", Path(__file__).resolve().parents[3] / "hardware" / "gateway_node"))


def _random_student(size: int = 64, classes: int = 7, name: str = "mobilenetv2_035"):
    torch.manual_seed(0)
    model = build_student(name, num_classes=classes, pretrained=False).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.3, 0.3)
            m.running_var.uniform_(0.3, 2.0)
            m.weight.data.uniform_(0.4, 1.6)
            m.bias.data.uniform_(-0.3, 0.3)
    return model


def _quantized(size: int = 64, classes: int = 7):
    model = _random_student(size, classes)
    g = int8.extract_graph(model, size)
    in_qp = int8.input_qparams(MEAN, STD)
    lut = int8.make_input_lut(MEAN, STD, in_qp)
    rng = np.random.default_rng(0)

    def batches():
        for _ in range(3):
            x = rng.integers(0, 256, (8, size, size, 3)).astype(np.uint8)
            xq = np.stack([lut[c][x[..., c]] for c in range(3)], -1).astype(np.int8)
            yield ((torch.from_numpy(xq).double() - in_qp.zp) * in_qp.scale).permute(0, 3, 1, 2)

    qm = int8.quantize(g, int8.calibrate(g, batches()), MEAN, STD)
    return model, g, qm, lut


def test_folded_graph_matches_pytorch_model():
    model = _random_student()
    g = int8.extract_graph(model, 64)
    assert int8.verify_graph(g, model, 64) < 1e-9


def test_unsupported_student_is_rejected():
    with pytest.raises(ValueError):
        build_student("mobilenetv3_small_100", num_classes=3, pretrained=False)


def test_quantize_multiplier_round_trips():
    for m in (1e-6, 3.3e-4, 0.0123, 0.5, 0.99999, 1.0, 1.75, 37.0):
        m0, sh = int8.quantize_multiplier(m)
        assert 2**30 <= m0 < 2**31 and 1 <= sh <= 62
        assert abs(m0 / 2**sh - m) / m < 1e-9


def test_requant_is_round_half_up_for_negative_values():
    acc = torch.tensor([-7, -6, -5, -1, 0, 1, 5, 6, 7], dtype=torch.int64)
    m0, sh = int8.quantize_multiplier(0.5)
    got = int8._requant(acc, torch.tensor(m0), torch.tensor(sh)).tolist()
    want = [(a * m0 + (1 << (sh - 1))) >> sh for a in acc.tolist()]   # Python ints: arithmetic shift
    assert got == want


def test_int8_model_tracks_float_model():
    model, g, qm, lut = _quantized()
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, (6, 64, 64, 3)).astype(np.uint8)
    xq = torch.from_numpy(np.stack([lut[c][x[..., c]] for c in range(3)], -1).astype(np.int8))
    lq = int8.logits_from_acc(qm, int8.int_forward(qm, xq)).numpy().ravel()
    lf = int8.float_forward(g, int8.dequantize_input(qm, xq).double()).numpy().ravel()
    assert np.corrcoef(lq, lf)[0, 1] > 0.95


def test_box_resize_matches_naive_box_mean():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (37, 53, 3)).astype(np.uint8)
    out = box_resize(img, 16, 11)
    x0, x1 = box_bounds(53, 16)
    y0, y1 = box_bounds(37, 11)
    for oy in range(11):
        for ox in range(16):
            box = img[y0[oy]:y1[oy], x0[ox]:x1[ox]].astype(np.int64)
            s, n = box.sum(axis=(0, 1)), box.shape[0] * box.shape[1]
            assert (out[oy, ox] == (s + n // 2) // n).all()


def test_box_bounds_tile_the_source_when_shrinking():
    for src, dst in ((640, 160), (643, 160), (480, 160), (1600, 160), (161, 160)):
        lo, hi = box_bounds(src, dst)
        assert lo[0] == 0 and hi[-1] == src and (hi[:-1] == lo[1:]).all()
        # the device maps each source pixel straight to its box
        o = ((np.arange(src) + 1) * dst - 1) // src
        assert ((lo[o] <= np.arange(src)) & (np.arange(src) < hi[o])).all()


def test_arena_plan_never_overlaps_live_tensors():
    _, g, qm, _ = _quantized()
    offsets, total = int8.plan_arena(qm)
    born = {g.input_id: -1}
    dies = {g.input_id: -1}
    for i, layer in enumerate(qm.layers):
        born[layer.out] = i
        dies.setdefault(layer.out, i)
        for t in (layer.inp, layer.inp2):
            if t >= 0:
                dies[t] = max(dies[t], i)
    dies[g.output_id] = len(qm.layers)
    ids = list(offsets)
    for a in ids:
        for b in ids:
            if a < b and not (dies[a] < born[b] or dies[b] < born[a]):
                ra = range(offsets[a], offsets[a] + g.t(a).nbytes)
                rb = range(offsets[b], offsets[b] + g.t(b).nbytes)
                assert not set(ra) & set(rb), (a, b)
    assert total >= max(offsets[t] + g.t(t).nbytes for t in ids)


def test_emit_c_writes_both_files(tmp_path):
    _, _, qm, _ = _quantized()
    info = int8.emit_c(qm, [f"Crop_disease_{i}" for i in range(7)], tmp_path, student="mobilenetv2_035")
    src = (tmp_path / "kn_edge_model.cpp").read_text()
    assert info["model_id"] in (tmp_path / "kn_edge_model.h").read_text()
    assert f"KN_EDGE_FORMAT_VERSION == {int8.FORMAT_VERSION}" in src
    assert src.count("// ") >= len(qm.layers)


def test_labels_that_would_break_json_are_rejected():
    with pytest.raises(ValueError):
        int8.check_labels(['Tomato "late" blight'])


@pytest.mark.skipif(shutil.which("g++") is None or not (GATEWAY_DIR / "kn_edge.cpp").exists(),
                    reason="needs g++ and the gateway sketch folder")
def test_edge_c_engine_matches_reference(tmp_path):
    _, _, qm, _ = _quantized()
    int8.emit_c(qm, [f"Crop_disease_{i}" for i in range(7)], tmp_path, student="mobilenetv2_035")
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include <stdio.h>\n#include "kn_edge.h"\n#include "kn_edge_model.h"\n'
        "int main(){ if (kn_edge_begin(&kn_edge_model)) return 2; uint32_t ms; int bad;\n"
        ' int r = kn_edge_self_test(&ms, &bad); printf("%d %d\\n", r, bad); return r ? 1 : 0; }\n'
    )
    exe = tmp_path / "edge_selftest"
    subprocess.run(["g++", "-std=gnu++17", "-O1", f"-I{GATEWAY_DIR}", f"-I{tmp_path}", str(main),
                    str(GATEWAY_DIR / "kn_edge.cpp"), str(GATEWAY_DIR / "kn_jpeg.cpp"),
                    str(tmp_path / "kn_edge_model.cpp"), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


# ---------------------------------------------------------------------------
# End to end on a tiny synthetic dataset: fake teacher -> train_edge -> export_edge
# ---------------------------------------------------------------------------

def _make_dataset(root: Path, classes=("Leaf_blight", "Leaf_healthy", "Leaf_rust"), per_class=12):
    import cv2
    import pandas as pd

    rng = np.random.default_rng(3)
    rows = []
    for ci, name in enumerate(classes):
        d = root / "raw" / name
        d.mkdir(parents=True)
        for i in range(per_class):
            img = np.zeros((120, 150, 3), np.uint8)
            img[..., ci] = 160 + rng.integers(0, 60)
            img = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
            p = d / f"{i}.jpg"
            cv2.imwrite(str(p), img)
            rows.append({"filepath": str(p), "class_name": name})
    df = pd.DataFrame(rows)
    splits = root / "splits"
    splits.mkdir()
    for split, frac in (("train", slice(0, 8)), ("val", slice(8, 10)), ("test", slice(10, 12))):
        part =pd.concat([g.iloc[frac] for _, g in df.groupby("class_name")])
        part.to_csv(splits / f"manifest_{split}.csv", index=False)
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
    (splits / "class_to_idx.json").write_text(json.dumps(class_to_idx))
    return splits, class_to_idx


def test_train_and_export_end_to_end(tmp_path):
    from configs.schema import Config
    from models.convnext_v2 import build_model
    from utils.checkpoint import save_checkpoint

    splits, class_to_idx = _make_dataset(tmp_path)
    cfg = Config()
    cfg.data.splits_dir = str(splits)
    cfg.data.image_size = 96
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.model.name = "mobilenetv2_035"
    cfg.model.pretrained = False
    cfg.log.log_dir = str(tmp_path / "logs")
    cfg.log.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.log.output_dir = str(tmp_path / "outputs")
    teacher = build_model(len(class_to_idx), cfg.model)
    teacher_path = tmp_path / "teacher.pt"
    save_checkpoint(teacher_path, model=teacher, epoch=0, best_metric=0.5, class_to_idx=class_to_idx, config=cfg)

    from edge import export_edge, train_edge

    train_edge.main(["--teacher", str(teacher_path), "--img-size", "64", "--epochs", "1", "--batch-size", "8",
                     "--num-workers", "0", "--device", "cpu", "--no-pretrained", "--student", "mobilenetv2_035",
                     "--name", "edge_test"])
    best = tmp_path / "ckpt" / "edge_test" / "best.pt"
    assert best.exists()

    gw = tmp_path / "gateway"
    gw.mkdir()
    export_edge.main(["--checkpoint", str(best), "--gateway-dir", str(gw), "--calib-images", "16",
                      "--batch-size", "4", "--num-workers", "0", "--device", "cpu"])
    assert (gw / "kn_edge_model.cpp").exists() and (gw / "kn_edge_model.h").exists()
    report = json.loads((tmp_path / "outputs" / "edge_test" / "edge_report.json").read_text())
    assert report["test"]["num_test_images"] == 6
    assert 0.0 <= report["test"]["int8_device_input"]["top1_accuracy"] <= 1.0
    assert (tmp_path / "outputs" / "edge_test" / "edge_int8.pt").exists()


def test_students_without_imagenet_weights_need_no_pretrained():
    with pytest.raises(ValueError, match="no ImageNet weights"):
        build_student("mobilenetv2_075", num_classes=3, pretrained=True)
    assert build_student("mobilenetv2_075", num_classes=3, pretrained=False) is not None


def test_train_and_export_from_unlabelled_photos(tmp_path):
    """No labelled dataset: the teacher labels a plain folder of photos."""
    import cv2

    from configs.schema import Config
    from models.convnext_v2 import build_model
    from utils.checkpoint import save_checkpoint

    photos = tmp_path / "leaf_photos" / "phone"
    photos.mkdir(parents=True)
    rng = np.random.default_rng(4)
    for i in range(220):   # train_edge wants at least 200
        img = np.clip(rng.normal(120, 40, (48, 64, 3)), 0, 255).astype(np.uint8)
        cv2.imwrite(str(photos / f"leaf_{i}.jpg"), img)
    (photos / "notes.txt").write_text("not a photo")
    (photos / "broken.jpg").write_bytes(b"not a jpeg")

    cfg = Config()
    cfg.data.splits_dir = str(tmp_path / "no_splits")   # must not be needed
    cfg.data.image_size = 96
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.model.name = "mobilenetv2_035"
    cfg.model.pretrained = False
    cfg.log.log_dir = str(tmp_path / "logs")
    cfg.log.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.log.output_dir = str(tmp_path / "outputs")
    class_to_idx = {"Leaf_blight": 0, "Leaf_healthy": 1, "Leaf_rust": 2}
    teacher = build_model(len(class_to_idx), cfg.model)
    teacher_path = tmp_path / "teacher.pt"
    save_checkpoint(teacher_path, model=teacher, epoch=0, best_metric=0.5, class_to_idx=class_to_idx, config=cfg)

    from edge import export_edge, train_edge
    from edge.unlabeled import HOLDOUT_LIST, TRAIN_LIST, read_list

    train_edge.main(["--teacher", str(teacher_path), "--images", str(tmp_path / "leaf_photos"),
                     "--img-size", "64", "--epochs", "1", "--batch-size", "8", "--limit-train", "32",
                     "--num-workers", "0", "--device", "cpu", "--no-pretrained", "--student", "mobilenetv2_035",
                     "--name", "edge_photos"])
    run = tmp_path / "ckpt" / "edge_photos"
    train_list, hold_list = read_list(run / TRAIN_LIST), read_list(run / HOLDOUT_LIST)
    assert len(train_list) + len(hold_list) == 221            # .txt skipped, broken .jpg kept (skipped when read)
    assert not set(train_list) & set(hold_list)
    assert (run / "best.pt").exists()

    gw = tmp_path / "gateway"
    gw.mkdir()
    export_edge.main(["--checkpoint", str(run / "best.pt"), "--gateway-dir", str(gw), "--calib-images", "16",
                      "--test-limit", "12", "--batch-size", "4", "--num-workers", "0", "--device", "cpu"])
    report = json.loads((tmp_path / "outputs" / "edge_photos" / "edge_report.json").read_text())
    agreement = report["holdout_agreement_with_teacher"]
    assert 1 <= agreement["num_holdout_images"] <= 12
    assert 0.0 <= agreement["int8_device_input"]["agree_top1"] <= agreement["int8_device_input"]["agree_top3"] <= 1.0
    assert "accuracy not measured" in (gw / "kn_edge_model.cpp").read_text()


def test_images_mode_refuses_without_teacher(tmp_path):
    from edge import train_edge

    with pytest.raises(SystemExit, match="needs the teacher"):
        train_edge.main(["--images", str(tmp_path), "--no-teacher"])


def test_holdout_split_is_disjoint_and_stable():
    from edge.unlabeled import split_holdout

    paths = [f"p{i}.jpg" for i in range(1000)]
    a_train, a_hold = split_holdout(paths, 0.1, seed=42)
    b_train, b_hold = split_holdout(paths, 0.1, seed=42)
    assert a_hold == b_hold and len(a_hold) == 100
    assert sorted(a_train + a_hold) == sorted(paths)

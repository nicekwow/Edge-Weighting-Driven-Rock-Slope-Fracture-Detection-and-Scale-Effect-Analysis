"""Build the scale-comparison tables and figures from saved predictions."""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "05_评价结果" / "图13至16"
SOURCE = ROOT / "02_实验数据" / "三幅重建图像与标注"
PRED = HERE / "predictions"
FIG = ROOT / "01_论文与图表" / "图13至16"
FIG.mkdir(parents=True, exist_ok=True)
ROWS = json.loads((HERE / "per_image_metrics.json").read_text(encoding="utf-8"))
SID = ("a11", "a13", "a14")
NAME = {"a11": "Image a", "a13": "Image b", "a14": "Image c"}
COLORS = {"a11": "#0072B2", "a13": "#D55E00", "a14": "#009E73"}
BLUE = "#0072B2"
GOLD = "#E69F00"
RED = np.array([230, 30, 65], dtype=np.float32)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def row(sid, strategy, size):
    return next(x for x in ROWS if x["sid"] == sid and x["strategy"] == strategy and x["size_px"] == size)


def pred_path(sid, strategy, size):
    key = strategy.lower().replace(" ", "_").replace("-", "_")
    return PRED / f"{sid}__{key}__{size or 'full'}.png"


def image(sid):
    return np.asarray(Image.open(SOURCE / f"{sid}.png").convert("RGB"))


def reference(sid):
    return np.asarray(Image.open(SOURCE / f"{sid}_mask.png")) != 0


def prediction(sid, strategy, size):
    return np.asarray(Image.open(pred_path(sid, strategy, size))) != 0


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# Tables, retaining one result per image rather than a pooled metric.
with (HERE / "Table4_reference_area.csv").open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Image size (pixels)", "Reference area (%)"])
    for sid in SID:
        writer.writerow([NAME[sid], "2560 × 1440", f"{100 * reference(sid).mean():.2f}"])
with (HERE / "Table5_per_image_mIoU.csv").open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Strategy", "Size (px)", "Image a mIoU (%)", "Image b mIoU (%)", "Image c mIoU (%)"])
    for strategy, sizes in (("Native full", (0,)), ("Resize", (1024, 512, 256)),
                            ("Sliding-window detection", (1024, 512, 256))):
        for size in sizes:
            writer.writerow([strategy, "2560 × 1440" if size == 0 else size] +
                            [f"{row(sid, strategy, size)['miou_pct']:.2f}" for sid in SID])


# Figure 13: source images and reference annotations, complete field of view.
fig, axs = plt.subplots(3, 2, figsize=(8.2, 7.5))
fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.06, wspace=0.02, hspace=0.09)
for i, sid in enumerate(SID):
    src = image(sid)
    ref = reference(sid)
    over = src.astype(np.float32).copy()
    over[ref] = 0.5 * over[ref] + 0.5 * RED
    for j, shown in enumerate((src, over.astype(np.uint8))):
        ax = axs[i, j]
        ax.imshow(shown)
        ax.set_axis_off()
        if i == 0:
            ax.set_title("Reconstructed view" if j == 0 else "Reference annotation", pad=7)
    axs[i, 0].text(-0.045, 0.5, NAME[sid], rotation=90, va="center", ha="center",
                   transform=axs[i, 0].transAxes, fontweight="bold")
fig.legend(handles=[Patch(facecolor="#e61e41", label="Annotated fracture")],
           loc="lower center", bbox_to_anchor=(0.54, 0.005), frameon=False)
save(fig, "Fig13_reference_annotations")


# Figure 14: image scores, arithmetic means, and fresh single-pass timings.
fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.2))
fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.10, wspace=0.19, hspace=0.28)
x = np.arange(4)
xlabels = ["Native full", "1024", "512", "256"]
baseline = {sid: row(sid, "Native full", 0) for sid in SID}
for j, strategy in enumerate(("Resize", "Sliding-window detection")):
    ax = axs[0, j]
    for sid in SID:
        values = [baseline[sid]["miou_pct"]] + [row(sid, strategy, s)["miou_pct"] for s in (1024, 512, 256)]
        ax.plot(x, values, "o-", color=COLORS[sid], linewidth=1.6, markersize=4, label=NAME[sid])
        for xx, value in zip(x, values):
            offset = (-28 if sid == "a14" else -12 if sid == "a13" else 5) if xx == 0 else (-15 if sid == "a13" else 5)
            ax.annotate(f"{value:.2f}", (xx, value), xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=7.1, color=COLORS[sid])
    ax.set_title("(a) Whole-image resize" if j == 0 else "(b) Sliding-window detection", loc="left")
    ax.set_ylabel("Per-image mIoU (%)")
    ax.set_xticks(x, xlabels)
    ax.set_ylim(44, 69)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
axs[0, 0].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

for strategy, color, offset, label in (("Resize", BLUE, -0.17, "Whole-image resize"),
                                       ("Sliding-window detection", GOLD, 0.17, "Sliding-window detection")):
    means = [np.mean([baseline[sid]["miou_pct"] for sid in SID])] + [
        np.mean([row(sid, strategy, s)["miou_pct"] for sid in SID]) for s in (1024, 512, 256)]
    bars = axs[1, 0].bar(x + offset, means, width=0.32, color=color, label=label)
    for bar, value in zip(bars, means):
        axs[1, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.2f}",
                       ha="center", va="bottom", fontsize=7.1)
    times = [sum(baseline[sid]["time_s"] for sid in SID)] + [
        sum(row(sid, strategy, s)["time_s"] for sid in SID) for s in (1024, 512, 256)]
    axs[1, 1].plot(x, times, "o-", color=color, linewidth=1.6, markersize=4, label=label)
axs[1, 0].set_title("(c) Mean mIoU", loc="left")
axs[1, 0].set_ylabel("Mean mIoU (%)")
axs[1, 0].set_ylim(0, 72)
axs[1, 0].set_xticks(x, xlabels)
axs[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
axs[1, 1].set_title("(d) Inference time", loc="left")
axs[1, 1].set_ylabel("Total inference time (s)")
axs[1, 1].set_yscale("log")
axs[1, 1].set_xticks(x, xlabels)
axs[1, 1].legend(frameon=False, fontsize=8, loc="lower left")
axs[1, 1].grid(color="#dddddd", linewidth=0.5)
for ax in axs.flat:
    ax.set_xlabel("Long edge / window side (px)")
save(fig, "Fig14_accuracy_and_time")


# Figure 15: all three complete images, reference and error overlays.
def error_overlay(src, ref, pred):
    out = src.astype(np.float32).copy()
    tp = ref & pred
    fp = ~ref & pred
    fn = ref & ~pred
    for region, color in ((tp, [0, 158, 115]), (fp, [213, 94, 0]), (fn, [0, 114, 178])):
        out[region] = 0.35 * out[region] + 0.65 * np.asarray(color)
    return out.astype(np.uint8)


fig, axs = plt.subplots(6, 4, figsize=(11.5, 10.8))
fig.subplots_adjust(left=0.055, right=0.995, top=0.985, bottom=0.055, wspace=0.06, hspace=0.28)
for i, sid in enumerate(SID):
    src = image(sid)
    ref = reference(sid)
    for j, size in enumerate((1024, 512, 256), 1):
        for strategy, axis in (("Resize", axs[2 * i, j]),
                               ("Sliding-window detection", axs[2 * i + 1, j])):
            pred = prediction(sid, strategy, size)
            axis.imshow(error_overlay(src, ref, pred))
            axis.set_title((f"Resize {size} × {size * 9 // 16}" if strategy == "Resize" else
                            f"Sliding-window {size} × {size}") +
                           f"\nmIoU {row(sid, strategy, size)['miou_pct']:.2f}%", fontsize=8, pad=3)
    axs[2 * i, 0].imshow(src)
    axs[2 * i, 0].set_title("Original image", fontsize=8, pad=3)
    ref_view = np.full_like(src, 255)
    ref_view[ref] = 0
    axs[2 * i + 1, 0].imshow(ref_view)
    axs[2 * i + 1, 0].set_title("Reference", fontsize=8, pad=3)
    axs[2 * i, 0].text(-0.10, 0.5, NAME[sid], rotation=90, va="center", ha="center",
                        transform=axs[2 * i, 0].transAxes, fontweight="bold")
for ax in axs.flat:
    ax.set_axis_off()
fig.legend(handles=[Patch(facecolor="#009E73", label="True positive"),
                    Patch(facecolor="#D55E00", label="False positive"),
                    Patch(facecolor="#0072B2", label="False negative")],
           loc="lower center", bbox_to_anchor=(0.52, 0.005), frameon=False, ncol=3, fontsize=9)
save(fig, "Fig15_error_maps")


# Figure 16: reuse the previously verified orientation definition unchanged.
from orientation import branches, orientations
histograms = {}
rose_rows = []
rose_segments = []
for sid in SID:
    masks = [("reference", 0, reference(sid))] + [
        (strategy, size, prediction(sid, strategy, size))
        for strategy in ("Resize", "Sliding-window detection") for size in (1024, 512, 256)]
    for strategy, size, mask in masks:
        paths, total = branches(mask)
        angles, weights, hist, short, degenerate = orientations(paths)
        histograms[(sid, strategy, size)] = hist
        rose_segments.extend(dict(image=NAME[sid], strategy=strategy, size_px=size,
                                  segment_index=i, angle_deg=float(a), length_px=float(w))
                             for i, (a, w) in enumerate(zip(angles, weights)))
        rose_rows.append(dict(image=NAME[sid], strategy=strategy, size_px=size,
                              skeleton_length_px=total, orientation_length_px=float(weights.sum()),
                              retained_fraction_pct=100 * weights.sum() / total if total else 0,
                              segment_count=len(angles), excluded_short_length_px=short,
                              degenerate_length_px=degenerate))
for rec in rose_rows:
    h = histograms[(next(k for k, v in NAME.items() if v == rec["image"]), rec["strategy"], rec["size_px"])]
    ref = histograms[(next(k for k, v in NAME.items() if v == rec["image"]), "reference", 0)]
    rec["D_pct"] = 100 * float(np.minimum(h, ref).sum()) if h.sum() else None
with (HERE / "trace_orientation_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rose_rows[0].keys())
    writer.writeheader()
    writer.writerows(rose_rows)

bin_rows = [dict(image=NAME[sid], strategy=strategy, size_px=size,
                 bin_center_deg=i*10, length_fraction=float(value))
            for (sid,strategy,size),hist in histograms.items() for i,value in enumerate(hist)]
for name, data in (("rose_bins.csv", bin_rows), ("trace_segments.csv", rose_segments)):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

fig, axs = plt.subplots(6, 3, subplot_kw={"projection": "polar"}, figsize=(8.7, 12.6))
fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.10, wspace=0.24, hspace=0.72)
limit = math.ceil(max(float(h.max() * 100) for h in histograms.values()) / 10) * 10
theta = np.deg2rad(np.arange(36) * 10)
for i, sid in enumerate(SID):
    for k, strategy in enumerate(("Resize", "Sliding-window detection")):
        rr = 2 * i + k
        color = BLUE if k == 0 else GOLD
        for j, size in enumerate((1024, 512, 256)):
            ax = axs[rr, j]
            ref = np.tile(histograms[(sid, "reference", 0)], 2) * 100
            pred = np.tile(histograms[(sid, strategy, size)], 2) * 100
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.bar(theta, np.sqrt(pred), width=np.deg2rad(9.4), color=color,
                   alpha=0.85, zorder=2, linewidth=0.25, edgecolor=color)
            edges = np.deg2rad(np.arange(-5, 356, 10))
            ax.stairs(np.sqrt(ref), edges, baseline=None, color="#222222", linewidth=1, zorder=3)
            ax.plot([edges[-1], edges[-1]], np.sqrt([ref[-1], ref[0]]),
                    color="#222222", linewidth=1, zorder=3)
            ax.set_ylim(0, np.sqrt(limit))
            ticks = list(range(20, limit + 1, 30))
            ax.set_yticks(np.sqrt(ticks))
            ax.set_yticklabels([str(t) + "%" for t in ticks], fontsize=7, color="#555555")
            ax.set_rlabel_position(225)
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=8)
            ax.tick_params(pad=1)
            ax.grid(color="#c1c9cd", linewidth=0.5, alpha=0.7)
            ax.spines["polar"].set_color("#b3bcc1")
            ax.spines["polar"].set_linewidth(0.6)
            if rr == 0:
                ax.set_title(str(size) + " px", pad=15, fontweight="bold", fontsize=12)
            rec = next(r for r in rose_rows if r["image"] == NAME[sid] and
                       r["strategy"] == strategy and r["size_px"] == size)
            d = rec["D_pct"]
            ax.text(0.5, -0.35, f"D = {d:.1f}%" if d is not None else "D = N/A",
                    transform=ax.transAxes, ha="center", fontsize=9)
        pos = axs[rr, 0].get_position()
        fig.text(0.025, (pos.y0 + pos.y1) / 2,
                 NAME[sid] + "\n" + ("Resize" if k == 0 else "Sliding-window\ndetection"),
                 ha="left", va="center", fontsize=10.5, fontweight="bold")
fig.legend(handles=[Patch(facecolor="none", edgecolor="#222222", label="Reference"),
                    Patch(facecolor=BLUE, label="Whole-image resize"),
                    Patch(facecolor=GOLD, label="Sliding-window detection")],
           loc="lower center", bbox_to_anchor=(0.55, 0.019), ncol=3, frameon=False, fontsize=9)
fig.text(0.55, 0.01, "10° bins · length weighted · 0° = image vertical",
         ha="center", fontsize=9)
save(fig, "Fig16_trace_orientations")
print("Generated Tables 4–5 and Figs. 13–16")

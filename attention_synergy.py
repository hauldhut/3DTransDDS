import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from torch_geometric.data import DataLoader

from data.dataset import TestbedDataset
from model.gnn_model import GCNPointNetGateTransformer


def extract_attention_all_batches(model, loader1, loader2, device, batch_size):
    model.eval()
    results = []

    for batch_idx, (data1, data2) in enumerate(zip(loader1, loader2)):
        data1 = data1.to(device)
        data2 = data2.to(device)
        labels = data1.y.view(-1).long().cpu().numpy()
        batch_len = len(labels)
        attention_per_layer = []

        def make_hook():
            def hook_fn(module, hook_input, hook_output):
                src = hook_input[0]
                _, attn_w = module.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
                attention_per_layer.append(attn_w.detach().cpu())

            return hook_fn

        hooks = [layer.register_forward_hook(make_hook()) for layer in model.transformer_encoder.layers]
        with torch.no_grad():
            output = model(data1, data2)
            probs = F.softmax(output, dim=1)
            scores = probs[:, 1].cpu().numpy()
            preds = probs.argmax(dim=1).cpu().numpy()
        for h in hooks:
            h.remove()

        n_layers = len(attention_per_layer)
        for i in range(batch_len):
            record = {"label": int(labels[i]), "pred_score": float(scores[i]), "pred_label": int(preds[i])}
            for l in range(n_layers):
                record[f"attn_L{l + 1}"] = attention_per_layer[l][i].numpy()
            results.append(record)

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size}/{len(loader1.dataset)} samples...")

    n_syn = sum(r["label"] == 1 for r in results)
    n_non = len(results) - n_syn
    acc = sum(r["label"] == r["pred_label"] for r in results) / len(results)
    print(f"Collected {len(results)} samples | syn={n_syn}, non={n_non}, acc={acc:.4f}")
    return results


def compute_group_stats(results, layer):
    key = f"attn_L{layer}"
    all_arr = np.stack([r[key].mean(axis=0) for r in results])
    syn_arr = np.stack([r[key].mean(axis=0) for r in results if r["label"] == 1])
    non_arr = np.stack([r[key].mean(axis=0) for r in results if r["label"] == 0])
    return {"all": all_arr.mean(0), "syn": syn_arr.mean(0), "non": non_arr.mean(0)}


def plot_main_figure(results, n_layers, n_heads, save_path):
    layer = n_layers
    tokens = ["Drug 1", "Drug 2", "Cell Line"]
    s = compute_group_stats(results, layer)
    cmap_blue = LinearSegmentedColormap.from_list("blue", ["#FFFFFF", "#1565C0", "#0D47A1"])
    cmap_red = LinearSegmentedColormap.from_list("red", ["#FFFFFF", "#B71C1C"])
    cmap_div = LinearSegmentedColormap.from_list("div", ["#C62828", "#FFFFFF", "#1565C0"])

    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
    ax_all = fig.add_subplot(gs[0, 0])
    ax_syn = fig.add_subplot(gs[0, 1])
    ax_non = fig.add_subplot(gs[0, 2])
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1:])

    def draw_hm(ax, mat, title, cmap, vmin=0, vmax=1):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
        ax.set_title(title)
        for r in range(3):
            for c in range(3):
                v = mat[r, c]
                ax.text(c, r, f"{v:.3f}", ha="center", va="center", color="white" if v > 0.6 else "#222")
        plt.colorbar(im, ax=ax, shrink=0.82)

    n_all = len(results)
    n_syn = sum(r["label"] == 1 for r in results)
    n_non = n_all - n_syn
    draw_hm(ax_all, s["all"], f"All Samples (n={n_all})", cmap_blue)
    draw_hm(ax_syn, s["syn"], f"Synergistic (n={n_syn})", cmap_blue)
    draw_hm(ax_non, s["non"], f"Non-synergistic (n={n_non})", cmap_red)

    diff = s["syn"] - s["non"]
    vabs = max(abs(diff.min()), abs(diff.max())) + 1e-6
    draw_hm(ax_diff, diff, "Difference (Syn - Non)", cmap_div, vmin=-vabs, vmax=vabs)

    pairs = [("D1->D2", 0, 1), ("D2->D1", 1, 0), ("D1->Cell", 0, 2), ("D2->Cell", 1, 2), ("Cell->D1", 2, 0), ("Cell->D2", 2, 1)]
    x = np.arange(len(pairs))
    w = 0.3
    syn_vals, non_vals = [], []
    for _, qi, ki in pairs:
        sv = [r[f"attn_L{layer}"].mean(0)[qi, ki] for r in results if r["label"] == 1]
        nv = [r[f"attn_L{layer}"].mean(0)[qi, ki] for r in results if r["label"] == 0]
        syn_vals.append(np.mean(sv))
        non_vals.append(np.mean(nv))
    ax_bar.bar(x - w / 2, syn_vals, w, label="Synergistic", color="#1565C0")
    ax_bar.bar(x + w / 2, non_vals, w, label="Non-synergistic", color="#C62828")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([p[0] for p in pairs], rotation=10)
    ax_bar.set_ylabel("Attention Score")
    ax_bar.legend()
    ax_bar.set_title(f"Cross-token attention (Layer {layer}, {n_heads} heads)")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_head(results, n_layers, n_heads, save_path):
    layer = n_layers
    key = f"attn_L{layer}"
    tokens = ["D1", "D2", "Cell"]
    syn_heads = np.stack([r[key] for r in results if r["label"] == 1])
    non_heads = np.stack([r[key] for r in results if r["label"] == 0])
    syn_mean = syn_heads.mean(0)
    non_mean = non_heads.mean(0)
    cmap = LinearSegmentedColormap.from_list("h", ["#FFFFFF", "#0D47A1"])

    fig, axes = plt.subplots(2, n_heads, figsize=(5.5 * n_heads, 9))
    if n_heads == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for h in range(n_heads):
        for row, (mat, label) in enumerate([(syn_mean[h], "Synergistic"), (non_mean[h], "Non-synergistic")]):
            ax = axes[row, h]
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(tokens)
            ax.set_yticklabels(tokens)
            ax.set_title(f"Head {h + 1} | {label}")
            for r in range(3):
                for c in range(3):
                    v = mat[r, c]
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "#222")
            plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Per-head attention (Layer {layer})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_case_study_combined(syn_result, non_result, n_layers, save_path):
    tokens = ["Drug 1", "Drug 2", "Cell Line"]
    cmap_syn = LinearSegmentedColormap.from_list("syn", ["#FFFFFF", "#1565C0"])
    cmap_non = LinearSegmentedColormap.from_list("non", ["#FFFFFF", "#B71C1C"])
    fig, axes = plt.subplots(2, n_layers, figsize=(7 * n_layers, 12), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    if n_layers == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row_idx, (r, status, cmap, score) in enumerate(
        [(syn_result, "Synergistic", cmap_syn, syn_result["pred_score"]), (non_result, "Non-synergistic", cmap_non, non_result["pred_score"])]
    ):
        for l in range(n_layers):
            attn = r[f"attn_L{l + 1}"].mean(axis=0)
            ax = axes[row_idx, l]
            im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(tokens, rotation=10)
            ax.set_yticklabels(tokens)
            ax.set_title(f"Layer {l + 1} | {status}")
            for ri in range(3):
                for ci in range(3):
                    v = attn[ri, ci]
                    ax.text(ci, ri, f"{v:.3f}", ha="center", va="center", color="white" if v > 0.55 else "#222")
            plt.colorbar(im, ax=ax, shrink=0.82)
        axes[row_idx, 0].annotate(
            f"Predicted score: {score:.4f}",
            xy=(-0.32, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
        )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run XAI analysis on trained transformer model.")
    parser.add_argument("--model-path", required=True, help="Path to trained .model file")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datafile", default="new_labels_0_10")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to reconstruct test split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="data/result")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNPointNetGateTransformer().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {args.model_path}")

    drug1_data = TestbedDataset(root=args.data_root, dataset=f"{args.datafile}_3d_new_drug1")
    drug2_data = TestbedDataset(root=args.data_root, dataset=f"{args.datafile}_3d_new_drug2")
    length = len(drug1_data)
    pot = int(length / 5)
    random_num = random.sample(range(0, length), length)
    test_num = random_num[pot * args.fold : pot * (args.fold + 1)]
    drug1_test = drug1_data[test_num]
    drug2_test = drug2_data[test_num]
    loader1_test = DataLoader(drug1_test, batch_size=args.batch_size, shuffle=False)
    loader2_test = DataLoader(drug2_test, batch_size=args.batch_size, shuffle=False)

    print("[1/4] Extracting attention...")
    results = extract_attention_all_batches(model, loader1_test, loader2_test, device, args.batch_size)
    n_layers = len(model.transformer_encoder.layers)
    n_heads = results[0]["attn_L1"].shape[0]
    print(f"Transformer: {n_layers} layers, {n_heads} heads")

    print("[2/4] Plotting main figure...")
    plot_main_figure(results, n_layers, n_heads, os.path.join(args.output_dir, "Fig_XAI_Attention_Main.png"))

    print("[3/4] Plotting per-head figure...")
    plot_per_head(results, n_layers, n_heads, os.path.join(args.output_dir, "Fig_XAI_PerHead.png"))

    print("[4/4] Plotting case-study figure...")
    syn_sorted = sorted([(idx, r) for idx, r in enumerate(results) if r["label"] == 1], key=lambda x: x[1]["pred_score"], reverse=True)
    non_sorted = sorted([(idx, r) for idx, r in enumerate(results) if r["label"] == 0], key=lambda x: x[1]["pred_score"])
    if syn_sorted and non_sorted:
        _, best_syn = syn_sorted[0]
        _, best_non = non_sorted[0]
        plot_case_study_combined(
            best_syn,
            best_non,
            n_layers,
            os.path.join(args.output_dir, "Fig_XAI_CaseStudy_Combined.png"),
        )
    else:
        print("Not enough samples for case study.")


if __name__ == "__main__":
    main()

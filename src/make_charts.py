"""
Đọc benchmark_results.csv, vẽ:
1. Bar chart Accuracy (DUNG/MOT_PHAN/SAI) giữa RAG và LLM thuần
2. Line chart Precision@K / Recall@K theo K (chỉ có ở nhánh RAG vì baseline không retrieval)
3. Bar chart Latency trung bình
"""
import pandas as pd
import matplotlib.pyplot as plt
from src import config


def accuracy_rate(df, verdict_col):
    return (df[verdict_col] == "DUNG").mean()


def plot_accuracy(df):
    rag_acc = accuracy_rate(df, "rag_verdict")
    base_acc = accuracy_rate(df, "baseline_verdict")

    plt.figure(figsize=(5, 4))
    plt.bar(["LLM thuần", "RAG"], [base_acc * 100, rag_acc * 100], color=["#D85A30", "#1D9E75"])
    plt.ylabel("Accuracy (%)")
    plt.title("So sánh Accuracy: LLM thuần vs RAG")
    plt.ylim(0, 100)
    for i, v in enumerate([base_acc * 100, rag_acc * 100]):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(config.OUT_DIR / "chart_accuracy.png", dpi=150)
    plt.close()


def plot_precision_recall(df):
    ks = config.K_VALUES
    precisions = [df[f"precision@{k}"].mean() for k in ks]
    recalls = [df[f"recall@{k}"].mean() for k in ks]

    plt.figure(figsize=(5, 4))
    plt.plot(ks, precisions, marker="o", label="Precision@K")
    plt.plot(ks, recalls, marker="s", label="Recall@K")
    plt.xlabel("K")
    plt.ylabel("Điểm số")
    plt.title("Precision@K / Recall@K của retriever (RAG)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.OUT_DIR / "chart_precision_recall.png", dpi=150)
    plt.close()


def plot_latency(df):
    rag_lat = df["rag_latency"].mean()
    base_lat = df["baseline_latency"].mean()

    plt.figure(figsize=(5, 4))
    plt.bar(["LLM thuần", "RAG"], [base_lat, rag_lat], color=["#D85A30", "#1D9E75"])
    plt.ylabel("Latency trung bình (giây)")
    plt.title("So sánh Latency: LLM thuần vs RAG")
    for i, v in enumerate([base_lat, rag_lat]):
        plt.text(i, v + 0.02, f"{v:.2f}s", ha="center")
    plt.tight_layout()
    plt.savefig(config.OUT_DIR / "chart_latency.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(config.RESULTS_CSV)
    plot_accuracy(df)
    plot_precision_recall(df)
    plot_latency(df)
    print(f"Đã lưu 3 biểu đồ vào {config.OUT_DIR}")

"""
Regenerate the accuracy/AUC vs. round-count plot from results.csv.
Run this AFTER distinguisher.py has produced results.csv in the same folder.

Usage:
    python3 plot_results.py
"""

import csv
import matplotlib.pyplot as plt

rounds, accs, aucs = [], [], []
with open("results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rounds.append(int(row["rounds"]))
        accs.append(float(row["accuracy"]))
        aucs.append(float(row["auc"]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds, accs, marker="o", label="Test accuracy", color="#2563eb")
ax.plot(rounds, aucs, marker="s", label="ROC-AUC", color="#f97316", linestyle="--")
ax.axhline(0.5, color="gray", linestyle=":", label="Random guessing (0.5)")
ax.set_xlabel("Number of SM4 rounds")
ax.set_ylabel("Score")
ax.set_title("ML Distinguisher Accuracy vs. SM4 Round Count")
ax.set_ylim(0.4, 1.05)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("sm4_distinguisher_results.png", dpi=150)
print("Saved plot to sm4_distinguisher_results.png")

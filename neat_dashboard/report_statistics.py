import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

LABELS = ["WALKING", "STANDING", "SITTING", "LAYING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
WINDOW_SIZE = 128
THRESHOLD = 0.9 * WINDOW_SIZE  # 115.2

df = pd.read_csv("prediction_report_capped.csv")

def get_actual_label(row):
    for label in LABELS:
        if row.get(label, 0) >= THRESHOLD:
            return label
    return "NOISE"

df["actual"] = df.apply(get_actual_label, axis=1)

noise_rows = df[df["actual"] == "NOISE"]
valid_rows = df[df["actual"] != "NOISE"]

correct = (valid_rows["actual"] == valid_rows["predicted"]).sum()
total_valid = len(valid_rows)
accuracy = (correct / total_valid * 100) if total_valid > 0 else 0.0

print("=" * 50)
print("PREDICTION REPORT STATISTICS")
print("=" * 50)
print(f"Total windows     : {len(df)}")
print(f"Noise windows     : {len(noise_rows)}")
print(f"Valid windows     : {total_valid}")
print(f"Correctly guessed : {correct}")
print(f"Accuracy          : {accuracy:.2f}%")
print()

if total_valid > 0:
    print("Per-label Accuracy:")
    for label in LABELS:
        label_rows = valid_rows[valid_rows["actual"] == label]
        if len(label_rows) == 0:
            continue
        label_correct = (label_rows["predicted"] == label).sum()
        label_acc = label_correct / len(label_rows) * 100
        print(f"  {label:<22}: {label_correct:>3}/{len(label_rows):<3} ({label_acc:.1f}%)")
    print()

if total_valid > 0:
    all_labels = sorted(set(valid_rows["actual"].tolist() + valid_rows["predicted"].tolist()))
    cm = confusion_matrix(valid_rows["actual"], valid_rows["predicted"], labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())
    print()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix plot saved to confusion_matrix.png")

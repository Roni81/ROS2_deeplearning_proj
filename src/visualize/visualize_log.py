import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "record_train.txt")
CSV_PATH = os.path.join(BASE_DIR, "training_log.csv")
PLOT_PATH = os.path.join(BASE_DIR, "training_plot.png")

print(f"Reading log from: {LOG_PATH}")

data = []
pattern = re.compile(r"\[Ep (\d+)\] Loss:([\d\.]+) \| Tr_Acc:([\d\.]+)% \| Val_Acc:([\d\.]+)%")

if not os.path.exists(LOG_PATH):
    print(f"Error: Log file not found at {LOG_PATH}")
else:
    with open(LOG_PATH, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data.append({
                    "Epoch": int(match.group(1)),
                    "Loss": float(match.group(2)),
                    "Train_Acc": float(match.group(3)),
                    "Val_Acc": float(match.group(4))
                })

if data:
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV to {CSV_PATH}")
    
    # Calculate Max Accuracy
    max_train_idx = df['Train_Acc'].idxmax()
    max_val_idx = df['Val_Acc'].idxmax()
    
    max_train_row = df.loc[max_train_idx]
    max_val_row = df.loc[max_val_idx]
    
    print(f"\n[Max Train Acc] Epoch: {int(max_train_row['Epoch'])}, Acc: {max_train_row['Train_Acc']}%")
    print(f"[Max Val Acc] Epoch: {int(max_val_row['Epoch'])}, Acc: {max_val_row['Val_Acc']}%")

    # Plotting
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Loss'], label='Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Train_Acc'], label='Train Acc', color='blue')
    plt.plot(df['Epoch'], df['Val_Acc'], label='Val Acc', color='green')
    
    # Max Text
    plt.scatter(max_train_row['Epoch'], max_train_row['Train_Acc'], color='blue')
    plt.text(max_train_row['Epoch'], max_train_row['Train_Acc'], 
             f"Max Tr: {max_train_row['Train_Acc']}% (Ep{int(max_train_row['Epoch'])})", 
             fontsize=9, ha='right')
             
    plt.scatter(max_val_row['Epoch'], max_val_row['Val_Acc'], color='green')
    plt.text(max_val_row['Epoch'], max_val_row['Val_Acc'], 
             f"Max Val: {max_val_row['Val_Acc']}% (Ep{int(max_val_row['Epoch'])})", 
             fontsize=9, ha='right')

    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}")
else:
    print("No data found matching the pattern.")

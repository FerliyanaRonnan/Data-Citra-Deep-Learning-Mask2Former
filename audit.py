import json

with open("Mask2Form-Pseudo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
total_lines = sum(len(c["source"]) for c in code_cells)
full_src = "\n".join("".join(c["source"]) for c in code_cells)

print(f"Total code cells: {len(code_cells)}")
print(f"Total code lines: {total_lines}")
print()

checks = [
    ("best_miou residual (self.best_miou)", "self.best_miou"),
    ("scheduler.load_state_dict residual", "scheduler.load_state_dict(checkpoint"),
    ("Emojis residual", "\u2705"),
    ("Hardcoded Colab paths", "/content/drive/MyDrive"),
    ("RARE_CLASSES defined/used", "RARE_CLASSES"),
    ("PseudoLabelDataset defined", "class PseudoLabelDataset"),
    ("CombinedDatasetWrapper defined", "class CombinedDatasetWrapper"),
    ("remove_small_components defined", "def remove_small_components"),
    ("Grid search ensemble weight", "weights_to_test"),
    ("Temperature scaling (0.5)", "temperature = 0.5"),
    ("np.rot90 optimization", "np.rot90"),
    ("BalancedBatchSampler in pseudo", "BalancedBatchSampler(combined_dataset"),
    ("Class distribution logging", "class_counts"),
    ("Multi-iteration pseudo loop", "NUM_ITERATIONS"),
    ("current_ckpt_path chain", "current_ckpt_path"),
    ("Dice in model selection", "val_dice"),
    ("Margin-based threshold", "margin = probs"),
    ("GridDistortion augmentation", "GridDistortion"),
    ("CoarseDropout augmentation", "CoarseDropout"),
    ("Worker seed (reproducibility)", "worker_init_fn"),
    ("EMA model", "EMA"),
    ("Experiment tracking (W&B)", "wandb"),
    ("argparse / config file", "argparse"),
]

print("Feature / Issue Audit:")
for label, pattern in checks:
    found = pattern in full_src
    status = "[YES]" if found else "[NO] "
    print(f"  {status} {label}")

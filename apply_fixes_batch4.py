import json

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def fix_cell(src):
    # ================================================================
    # FIX 1: best_miou -> best_score (konsisten di seluruh notebook)
    # ================================================================
    src = src.replace(
        '        self.best_miou = 0.0\n        self.best_smoothed_val_loss',
        '        self.best_score = 0.0\n        self.best_smoothed_val_loss'
    )
    src = src.replace(
        '            val_score = (val_miou + val_dice) / 2.0\n            if val_score > self.best_miou:\n                self.best_miou = val_score\n                torch.save(checkpoint, self.best_checkpoint_path)\n                print(f"  New best Score (mIoU+Dice)/2: {val_score:.4f}")',
        '            val_score = (val_miou + val_dice) / 2.0\n            if val_score > self.best_score:\n                self.best_score = val_score\n                torch.save(checkpoint, self.best_checkpoint_path)\n                print(f"  New best Score (mIoU+Dice)/2: {val_score:.4f}")'
    )
    src = src.replace(
        "                'best_miou': self.best_miou,\n                'best_smoothed_val_loss': self.best_smoothed_val_loss,",
        "                'best_score': self.best_score,\n                'best_smoothed_val_loss': self.best_smoothed_val_loss,"
    )
    src = src.replace(
        '        print(f"\\nBest Score: {self.best_miou:.4f}")',
        '        print(f"\\nBest Score: {self.best_score:.4f}")'
    )

    # FIX 1b: load_m2f() display
    src = src.replace(
        "print(f\"[OK] Loaded from {ckpt_path} | epoch {ckpt.get('epoch', '?')} | best mIoU {ckpt.get('best_miou', 0):.4f}\")",
        "print(f\"[OK] Loaded from {ckpt_path} | epoch {ckpt.get('epoch', '?')} | best score {ckpt.get('best_score', ckpt.get('best_miou', 0)):.4f}\")"
    )
    src = src.replace(
        "print(f\"[OK] SegFormer loaded | epoch {ckpt.get('epoch', '?')} | best mIoU {ckpt.get('best_miou', 0):.4f}\")",
        "print(f\"[OK] SegFormer loaded | epoch {ckpt.get('epoch', '?')} | best score {ckpt.get('best_score', ckpt.get('best_miou', 0)):.4f}\")"
    )

    # FIX 1c: Cell 14 - Baseline trainer restore
    src = src.replace(
        "    trainer.best_miou = checkpoint.get('best_miou', 0.0)\n    trainer.best_smoothed_val_loss = checkpoint.get('best_smoothed_val_loss', float('inf'))\n    trainer.patience_counter = checkpoint.get('patience_counter', 0)\n    trainer.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n    print(f\"[OK] Trainer state restored (best_miou: {trainer.best_miou:.4f})\")",
        "    trainer.best_score = checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))\n    trainer.best_smoothed_val_loss = checkpoint.get('best_smoothed_val_loss', float('inf'))\n    trainer.patience_counter = checkpoint.get('patience_counter', 0)\n    trainer.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n    print(f\"[OK] Trainer state restored (best_score: {trainer.best_score:.4f})\")"
    )
    src = src.replace(
        "    best_miou = checkpoint.get('best_miou', 0.0)\n    print(f\"   Resume from epoch {start_epoch-1}, best mIoU: {best_miou:.4f}\")\nelse:\n    print(\"[FAIL] No checkpoint found, training from scratch\")\n    best_miou = 0.0",
        "    best_score = checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))\n    print(f\"   Resume from epoch {start_epoch-1}, best score: {best_score:.4f}\")\nelse:\n    print(\"[FAIL] No checkpoint found, training from scratch\")\n    best_score = 0.0"
    )
    src = src.replace(
        'print(f"\\n[OK] Training completed! Best mIoU: {trainer.best_miou:.4f}")',
        'print(f"\\n[OK] Training completed! Best score: {trainer.best_score:.4f}")'
    )

    # FIX 1d: Cell 15 baseline inference
    src = src.replace(
        "print(f\"Loaded from epoch {ckpt['epoch']}, best mIoU: {ckpt['best_miou']:.4f}\")",
        "print(f\"Loaded from epoch {ckpt['epoch']}, best score: {ckpt.get('best_score', ckpt.get('best_miou', 0)):.4f}\")"
    )

    # FIX 1e: Task 1 ensemble prints
    src = src.replace(
        "print(f\"Mask2Former best Score: {ckpt_m2f.get('best_miou', 0):.4f}\" if ckpt_m2f else \"Mask2Former loaded\")\nprint(f\"SegFormer best Score: {ckpt_seg.get('best_miou', 0):.4f}\" if ckpt_seg else \"SegFormer loaded\")",
        "print(f\"Mask2Former best Score: {ckpt_m2f.get('best_score', ckpt_m2f.get('best_miou', 0)):.4f}\" if ckpt_m2f else \"Mask2Former loaded\")\nprint(f\"SegFormer best Score: {ckpt_seg.get('best_score', ckpt_seg.get('best_miou', 0)):.4f}\" if ckpt_seg else \"SegFormer loaded\")"
    )

    # FIX 1f: Task 2 finetune
    src = src.replace(
        "if checkpoint is not None:\n    trainer.best_miou = checkpoint.get('best_miou', 0.0)\n    trainer.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n    print(f\"[OK] Trainer state restored\")",
        "if checkpoint is not None:\n    trainer.best_score = checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))\n    trainer.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n    print(f\"[OK] Trainer state restored\")"
    )
    src = src.replace(
        "    'best_miou': trainer.best_miou,\n    'history': trainer.history,\n}, CKPT_M2F_FINETUNE)",
        "    'best_score': trainer.best_score,\n    'history': trainer.history,\n}, CKPT_M2F_FINETUNE)"
    )

    # FIX 1g: Task 3 weighted
    src = src.replace(
        "if checkpoint is not None:\n    trainer.best_miou = checkpoint.get('best_miou', 0.0)\n    trainer.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n\n# ========== TRAIN ==========\nprint(f\"\\nStarting weighted training",
        "if checkpoint is not None:\n    trainer.best_score = checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))\n    trainer.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n\n# ========== TRAIN ==========\nprint(f\"\\nStarting weighted training"
    )
    src = src.replace(
        "    'best_miou': trainer.best_miou,\n    'history': trainer.history,\n}, CKPT_M2F_WEIGHTED)",
        "    'best_score': trainer.best_score,\n    'history': trainer.history,\n}, CKPT_M2F_WEIGHTED)"
    )

    # FIX 1h: retrain_with_pseudo_labels trainer restore
    src = src.replace(
        "        trainer_ps.best_miou = checkpoint.get('best_score', 0.0)\n        trainer_ps.best_smoothed_val_loss = checkpoint.get('best_smoothed_val_loss', float('inf'))\n        trainer_ps.patience_counter = checkpoint.get('patience_counter', 0)\n        trainer_ps.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n        print(f\"[OK] Trainer state restored (best_score: {trainer_ps.best_miou:.4f})\")",
        "        trainer_ps.best_score = checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))\n        trainer_ps.best_smoothed_val_loss = checkpoint.get('best_smoothed_val_loss', float('inf'))\n        trainer_ps.patience_counter = checkpoint.get('patience_counter', 0)\n        trainer_ps.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n        print(f\"[OK] Trainer state restored (best_score: {trainer_ps.best_score:.4f})\")"
    )

    # ================================================================
    # FIX 2: Scheduler safe resume - fast-forward instead of load_state_dict
    # ================================================================
    # Baseline training (Cell 14)
    src = src.replace(
        "    try:\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])\n        print(\"[OK] Optimizer & scheduler state loaded\")\n    except Exception as e:\n        print(f\"[WARN] Could not load optimizer state: {e}\")\n\n# ========== 7. LOSS FUNCTION",
        "    try:\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        print(\"[OK] Optimizer state loaded\")\n    except Exception as e:\n        print(f\"[WARN] Could not load optimizer state: {e}\")\n    steps_to_catch_up = (start_epoch - 1) * (len(train_loader) // TRAIN_CONFIG['accumulation_steps'])\n    for _ in range(steps_to_catch_up):\n        scheduler.step()\n    print(f\"[OK] Scheduler fast-forwarded {steps_to_catch_up} steps to epoch {start_epoch-1}\")\n\n# ========== 7. LOSS FUNCTION"
    )
    # Task 2 finetune
    src = src.replace(
        "if checkpoint is not None:\n    try:\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])\n        print(\"[OK] Optimizer state loaded\")\n    except Exception as e:\n        print(f\"[WARN] Could not load optimizer state: {e}\")\n\n# ========== LOSS ==========\nloss_fn = FloodSegLoss(CLASS_WEIGHTS.to(device), lovasz_weight=0.75",
        "if checkpoint is not None:\n    try:\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        print(\"[OK] Optimizer state loaded\")\n    except Exception as e:\n        print(f\"[WARN] Could not load optimizer state: {e}\")\n    steps_to_catch_up = (start_epoch - 1) * (len(train_loader_ft) // TRAIN_CONFIG_FINETUNE['accumulation_steps'])\n    for _ in range(steps_to_catch_up):\n        scheduler.step()\n    print(f\"[OK] Scheduler fast-forwarded {steps_to_catch_up} steps\")\n\n# ========== LOSS ==========\nloss_fn = FloodSegLoss(CLASS_WEIGHTS.to(device), lovasz_weight=0.75"
    )
    # Task 3 weighted
    src = src.replace(
        "if checkpoint is not None:\n    try:\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])\n        print(\"[OK] Optimizer state loaded\")\n    except Exception as e:\n        print(f\"[WARN] Could not load optimizer state: {e}\")\n\n# ========== LOSS WITH ENHANCED WEIGHTS",
        "if checkpoint is not None:\n    try:\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        print(\"[OK] Optimizer state loaded\")\n    except Exception as e:\n        print(f\"[WARN] Could not load optimizer state: {e}\")\n    steps_to_catch_up = (start_epoch - 1) * (len(train_loader_wt) // 8)\n    for _ in range(steps_to_catch_up):\n        scheduler.step()\n    print(f\"[OK] Scheduler fast-forwarded {steps_to_catch_up} steps\")\n\n# ========== LOSS WITH ENHANCED WEIGHTS"
    )
    # retrain_with_pseudo_labels
    src = src.replace(
        "    if checkpoint is not None:\n        try:\n            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])\n            print(\"[OK] Optimizer state loaded\")\n        except Exception as e:\n            print(f\"[WARN] Could not load optimizer state: {e}\")\n            \n    loss_fn_ps",
        "    if checkpoint is not None:\n        try:\n            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n            print(\"[OK] Optimizer state loaded\")\n        except Exception as e:\n            print(f\"[WARN] Could not load optimizer state: {e}\")\n        steps_to_catch_up = (start_epoch - 1) * (len(combined_loader) // config['accumulation_steps'])\n        for _ in range(steps_to_catch_up):\n            scheduler.step()\n        print(f\"[OK] Scheduler fast-forwarded {steps_to_catch_up} steps\")\n\n    loss_fn_ps"
    )

    # ================================================================
    # FIX 3: Pseudo-label class distribution logging (guard duplicate)
    # ================================================================
    if "class_counts = {c: 0 for c in range(NUM_CLASSES)}" not in src:
        src = src.replace(
            "    accepted = 0\n    total_pix = 0\n    conf_pix = 0",
            "    accepted = 0\n    total_pix = 0\n    conf_pix = 0\n    class_counts = {c: 0 for c in range(NUM_CLASSES)}"
        )
        src = src.replace(
            "        total_pix += H * W\n        conf_pix += (pseudo_mask != 255).sum().item()\n        accepted += 1",
            "        total_pix += H * W\n        conf_pix += (pseudo_mask != 255).sum().item()\n        accepted += 1\n        for c in range(NUM_CLASSES):\n            class_counts[c] += (pseudo_mask == c).sum().item()"
        )
        src = src.replace(
            '    print(f"Generated {accepted} pseudo-masks")\n    print(f"Confident pixels (avg): {conf_pix/total_pix*100:.1f}%")\n    return accepted',
            '    print(f"Generated {accepted} pseudo-masks")\n    print(f"Confident pixels (avg): {conf_pix/total_pix*100:.1f}%")\n    print("\\nClass Distribution in Pseudo-Labels:")\n    total_valid = sum(class_counts.values())\n    for c in range(NUM_CLASSES):\n        if c in EMPTY_CLASSES:\n            continue\n        pct = (class_counts[c] / max(1, total_valid)) * 100\n        flag = " [LOW - check confirmation bias]" if pct < 0.1 else ""\n        print(f"  Class {c} ({CLASS_NAMES[c]}): {pct:.2f}% ({class_counts[c]} px){flag}")\n    return accepted'
        )

    return src

changes_made = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    original = ''.join(cell['source'])
    fixed = fix_cell(original)
    if fixed != original:
        cell['source'] = fixed.splitlines(keepends=True)
        changes_made.append(i)

print(f"Fixed cells: {changes_made}")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Done writing notebook")

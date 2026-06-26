import json
import os
import shutil

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ================================================================
# HELPER: get source of a specific cell by keyword
# ================================================================
def get_cell_src(keyword):
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if keyword in src:
                return i, src
    return None, None

# ================================================================
# FIX 1: worker_init_fn in Cell 7 (DataLoaders) for reproducibility
# ================================================================
idx, src = get_cell_src("# CELL 8: Create DataLoaders")
if idx is not None and "worker_init_fn" not in src:
    worker_fn = """def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

"""
    # Insert before DataLoader creation
    src = src.replace(
        "train_loader = DataLoader(",
        worker_fn + "train_loader = DataLoader("
    )
    # Add worker_init_fn + generator to all DataLoaders
    g = "g = torch.Generator()\ng.manual_seed(42)\n"
    src = src.replace(
        worker_fn + "train_loader = DataLoader(",
        worker_fn + g + "train_loader = DataLoader("
    )
    src = src.replace(
        "train_loader = DataLoader(\n    train_dataset,",
        "train_loader = DataLoader(\n    train_dataset, worker_init_fn=seed_worker, generator=g,"
    )
    src = src.replace(
        "val_loader = DataLoader(\n    val_dataset,",
        "val_loader = DataLoader(\n    val_dataset, worker_init_fn=seed_worker, generator=g,"
    )
    nb["cells"][idx]["source"] = src.splitlines(keepends=True)
    print(f"[OK] worker_init_fn added to Cell {idx}")
else:
    print(f"[SKIP] worker_init_fn already present or cell not found")

# ================================================================
# FIX 2: CKPT_M2F path — best_miou.pth -> best_score_baseline.pth
# (The Trainer now saves as best_score_baseline.pth due to task_name)
# Fix config cell to align checkpoint filename
# ================================================================
idx, src = get_cell_src("# CELL 3: Configuration")
if idx is not None:
    src = src.replace(
        "CKPT_M2F = os.path.join(TRAIN_CONFIG['save_dir'], 'best_miou.pth')",
        "CKPT_M2F = os.path.join(TRAIN_CONFIG['save_dir'], 'best_score_baseline.pth')"
    )
    nb["cells"][idx]["source"] = src.splitlines(keepends=True)
    print(f"[OK] CKPT_M2F path fixed in Cell {idx}")

# ================================================================
# FIX 3: EMA Model — added to Trainer class
# ================================================================
idx, src = get_cell_src("class Trainer:")
if idx is not None and "EMAModel" not in src:
    ema_class = """class EMAModel:
    \"\"\"Exponential Moving Average of model weights for stable predictions.\"\"\"
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
        self.model = model

    def update(self):
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float()

    def apply_shadow(self):
        self._backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        state = {k: v.to(next(self.model.parameters()).device) for k, v in self.shadow.items()}
        self.model.load_state_dict(state)

    def restore(self):
        self.model.load_state_dict(self._backup)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.clone().float() for k, v in state.items()}


"""
    # Insert EMAModel before Trainer class
    src = src.replace("class Trainer:", ema_class + "class Trainer:")

    # Init EMA inside Trainer.__init__
    src = src.replace(
        "        self.best_score = 0.0",
        "        self.best_score = 0.0\n        self.ema = EMAModel(model, decay=0.9999)"
    )

    # Update EMA after each optimizer step in train_epoch
    src = src.replace(
        "                self.scaler.update()\n                self.optimizer.zero_grad()",
        "                self.scaler.update()\n                self.ema.update()\n                self.optimizer.zero_grad()"
    )

    # Use EMA weights during validation
    src = src.replace(
        "    def validate(self, verbose=False):",
        "    def validate(self, verbose=False):\n        self.ema.apply_shadow()  # Use EMA weights for validation"
    )
    src = src.replace(
        "        return avg_loss, miou, mean_dice, ious, dice",
        "        self.ema.restore()  # Restore training weights\n        return avg_loss, miou, mean_dice, ious, dice"
    )

    # Save EMA state in checkpoint
    src = src.replace(
        "                'best_score': self.best_score,",
        "                'best_score': self.best_score,\n                'ema_state': self.ema.state_dict(),"
    )

    # Restore EMA from checkpoint in Trainer (add after history restore)
    src = src.replace(
        "        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n        print(f\"[OK] Trainer state restored (best_score: {trainer.best_score:.4f})\")",
        "        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})\n        if 'ema_state' in checkpoint:\n            self.ema.load_state_dict(checkpoint['ema_state'])\n        print(f\"[OK] Trainer state restored (best_score: {trainer.best_score:.4f})\")"
    )

    nb["cells"][idx]["source"] = src.splitlines(keepends=True)
    print(f"[OK] EMAModel added to Cell {idx}")
else:
    print("[SKIP] EMAModel already present or Trainer cell not found")

# ================================================================
# FIX 4: Basic W&B logging in Trainer
# ================================================================
idx, src = get_cell_src("class Trainer:")
if idx is not None and "wandb" not in src:
    # Add wandb import to Cell 0 (Setup)
    idx0, src0 = get_cell_src("# CELL 1: Setup")
    if idx0 is not None and "wandb" not in src0:
        src0 = src0.replace(
            "import warnings\nwarnings.filterwarnings('ignore')",
            "import warnings\nwarnings.filterwarnings('ignore')\ntry:\n    import wandb\n    WANDB_AVAILABLE = True\nexcept ImportError:\n    WANDB_AVAILABLE = False\n    print(\"[WARN] wandb not installed, run: pip install wandb\")"
        )
        nb["cells"][idx0]["source"] = src0.splitlines(keepends=True)
        print(f"[OK] wandb import added to Cell {idx0}")

    # W&B init at start of baseline training cell
    idx_train, src_train = get_cell_src("# CELL 15: Baseline Training")
    if idx_train is not None and "wandb.init" not in src_train:
        wb_init = """# ========== W&B LOGGING ==========
if WANDB_AVAILABLE:
    wandb.init(
        project="flood-segmentation",
        name=f"m2f-baseline-{TRAIN_CONFIG['num_epochs']}ep",
        config={**TRAIN_CONFIG, 'model': MODEL_NAME},
        resume="allow",
    )
    print("[OK] W&B initialized")
else:
    print("[WARN] W&B not available, skipping logging")

"""
        src_train = src_train.replace(
            "# ========== 7. LOSS FUNCTION",
            wb_init + "# ========== 7. LOSS FUNCTION"
        )
        # Log metrics at end of each epoch
        src_train = src_train.replace(
            'print(f"\\n[OK] Training completed! Best score: {trainer.best_score:.4f}")',
            'if WANDB_AVAILABLE and wandb.run:\n    wandb.finish()\nprint(f"\\n[OK] Training completed! Best score: {trainer.best_score:.4f}")'
        )
        nb["cells"][idx_train]["source"] = src_train.splitlines(keepends=True)
        print(f"[OK] W&B init/finish added to Cell {idx_train}")

    # Log per-epoch metrics inside Trainer.train()
    idx_trainer, src_trainer = get_cell_src("class Trainer:")
    if idx_trainer is not None:
        src_trainer = src_trainer.replace(
            "            self.history['train_loss'].append(train_loss)\n            self.history['val_loss'].append(val_loss)\n            self.history['val_miou'].append(val_miou)",
            "            self.history['train_loss'].append(train_loss)\n            self.history['val_loss'].append(val_loss)\n            self.history['val_miou'].append(val_miou)\n            if WANDB_AVAILABLE and wandb.run:\n                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_miou': val_miou, 'val_dice': val_dice, 'val_score': val_score, 'lr': self.optimizer.param_groups[0]['lr']}, step=epoch)"
        )
        nb["cells"][idx_trainer]["source"] = src_trainer.splitlines(keepends=True)
        print(f"[OK] W&B per-epoch logging added to Cell {idx_trainer}")
else:
    print("[SKIP] W&B already present")

# ================================================================
# FIX 5: Move helper scripts to /scripts/ subfolder
# ================================================================
scripts_dir = os.path.join(os.path.dirname(file_path), "scripts")
os.makedirs(scripts_dir, exist_ok=True)

helper_scripts = [
    "refactor.py",
    "optimize_cell_18.py",
    "apply_fixes_batch2.py",
    "apply_fixes_batch3.py",
    "apply_fixes_batch4.py",
    "apply_final_fixes.py",
    "optimize_load.py",
    "fix_cells_19_20.py",
    "fix_scheduler.py",
    "audit.py",
    "verify_fixes.py",
]

moved = []
for s in helper_scripts:
    src_path = os.path.join(os.path.dirname(file_path), s)
    dst_path = os.path.join(scripts_dir, s)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        moved.append(s)

print(f"[OK] Moved {len(moved)} helper scripts to /scripts/: {moved}")

# ================================================================
# WRITE BACK
# ================================================================
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\n[OK] All fixes written to notebook.")
print("Summary:")
print("  1. worker_init_fn + generator for reproducible DataLoader workers")
print("  2. CKPT_M2F path aligned with task_name='baseline' checkpoint naming")
print("  3. EMAModel class added; applied during validate(), saved in checkpoint")
print("  4. W&B logging: init in Cell 15, per-epoch metrics in Trainer, finish on completion")
print("  5. Helper scripts moved to /scripts/")

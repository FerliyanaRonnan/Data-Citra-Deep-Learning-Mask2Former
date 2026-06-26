import json

with open("Mask2Form-Pseudo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "seed_worker" not in src or "# CELL 8: Create DataLoaders" not in src:
        continue

    # Ensure generator is defined
    if "g = torch.Generator()" not in src:
        src = src.replace(
            "def seed_worker(worker_id):",
            "g = torch.Generator()\ng.manual_seed(42)\n\ndef seed_worker(worker_id):"
        )

    # Inject into train_loader (uses batch_sampler)
    src = src.replace(
        "train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True)",
        "train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)"
    )
    # Inject into val_loader
    src = src.replace(
        "val_loader   = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)",
        "val_loader   = DataLoader(val_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)"
    )
    # Catch any simple DataLoader(... num_workers without worker_init_fn
    import re
    # Add worker_init_fn to any remaining DataLoader with num_workers but not already patched
    src = re.sub(
        r"(DataLoader\([^)]*num_workers=num_workers, pin_memory=True\))",
        lambda m: m.group(0).replace(", pin_memory=True)", ", pin_memory=True, worker_init_fn=seed_worker, generator=g)") if "worker_init_fn" not in m.group(0) else m.group(0),
        src
    )

    cell["source"] = src.splitlines(keepends=True)
    print(f"[OK] worker_init_fn injected into Cell {i} DataLoaders")
    break

with open("Mask2Form-Pseudo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Update audit script to check seed_worker too
print("Done")

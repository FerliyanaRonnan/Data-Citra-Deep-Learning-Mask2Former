import json
import re

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = "".join(cell['source'])
    
    # --- ISSUE 1: best_miou to best_score ---
    if "class Trainer:" in src:
        src = src.replace("self.best_miou = 0.0", "self.best_score = 0.0")
        src = src.replace("val_score > self.best_miou", "val_score > self.best_score")
        src = src.replace("self.best_miou = val_score", "self.best_score = val_score")
        src = src.replace("self.best_miou:.4f", "self.best_score:.4f")
        src = src.replace("'best_miou': self.best_miou", "'best_score': self.best_score")
        src = src.replace("checkpoint.get('best_miou', 0.0)", "checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))")
    
    if "trainer.best_miou = checkpoint.get" in src:
        src = src.replace("trainer.best_miou = checkpoint.get('best_miou', 0.0)", "trainer.best_score = checkpoint.get('best_score', checkpoint.get('best_miou', 0.0))")
        src = src.replace("best_miou: {trainer.best_miou:.4f}", "best_score: {trainer.best_score:.4f}")
        
    if "trainer.best_miou" in src: # Catch-all for print statements outside classes
        src = src.replace("trainer.best_miou", "trainer.best_score")
        
    # Also in Pseudo-labeling retrain
    if "trainer_ps.best_miou" in src:
        src = src.replace("trainer_ps.best_miou", "trainer_ps.best_score")
        src = src.replace("best_miou: {trainer_ps.best_score", "best_score: {trainer_ps.best_score")
        src = src.replace("checkpoint.get('best_miou'", "checkpoint.get('best_score', checkpoint.get('best_miou', 0.0)")

    # --- ISSUE 3: Safe Scheduler Resume ---
    if "scheduler.load_state_dict(checkpoint['scheduler_state_dict'])" in src:
        # Extract loader name
        loader_name = "train_loader"
        m = re.search(r"len\(([a-zA-Z0-9_]+loader[a-zA-Z0-9_]*)\)", src)
        if m:
            loader_name = m.group(1)
            
        old_sched = "scheduler.load_state_dict(checkpoint['scheduler_state_dict'])"
        new_sched = f"""# scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # Removed due to total_steps mismatch risk
            # Fast-forward scheduler safely instead
            steps_to_catch_up = (start_epoch - 1) * len({loader_name})
            for _ in range(steps_to_catch_up):
                scheduler.step()"""
        src = src.replace(old_sched, new_sched)

    # --- ISSUE 2: Pseudo-Label Class Distribution ---
    if "def generate_pseudo_labels_optimized" in src:
        if "class_counts = {c: 0 for c in range(10)}" not in src:
            src = src.replace("conf_pix = 0", "conf_pix = 0\n    class_counts = {c: 0 for c in range(10)}")
            
            count_update = """        total_pix += H * W
        conf_pix += (pseudo_mask != 255).sum().item()
        
        for c in range(10):
            class_counts[c] += (pseudo_mask == c).sum().item()"""
            src = src.replace("total_pix += H * W\n        conf_pix += (pseudo_mask != 255).sum().item()", count_update)
            
            print_counts = """    print(f"Generated {accepted} pseudo-masks")
    print(f"Confident pixels (avg): {conf_pix/total_pix*100:.1f}%")
    print("\\nClass Distribution in Pseudo-Labels:")
    total_valid = sum(class_counts.values())
    for c in range(10):
        if c in EMPTY_CLASSES: continue
        pct = (class_counts[c] / max(1, total_valid)) * 100
        print(f"  Class {c} ({CLASS_NAMES[c]}): {pct:.2f}% ({class_counts[c]} px)")"""
            src = src.replace("print(f\"Generated {accepted} pseudo-masks\")\n    print(f\"Confident pixels (avg): {conf_pix/total_pix*100:.1f}%\")", print_counts)

    cell['source'] = src.splitlines(keepends=True)
    
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Batch 3 fixes applied successfully!")

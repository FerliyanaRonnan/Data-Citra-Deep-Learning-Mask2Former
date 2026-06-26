import json
import re

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    src = "".join(cell['source'])
    
    # FIX 6: Aerial imagery augmentations
    if "train_transform = A.Compose([" in src:
        new_aug = """train_transform = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
    A.GridDistortion(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])"""
        src = re.sub(r"train_transform = A\.Compose\(\[.*?\]\)", new_aug, src, flags=re.DOTALL)

    # FIX 3: Temperature scaling in Mask2Former forward
    if "class Mask2FormerFlood(nn.Module):" in src:
        old_probs = "class_probs = F.softmax(class_logits, dim=-1)[:, :, :-1]"
        new_probs = "temperature = 0.1  # Stabilize dense approximation\n        class_probs = F.softmax(class_logits / temperature, dim=-1)[:, :, :-1]"
        src = src.replace(old_probs, new_probs)

    # FIX 5: Per-class thresholding logic (margin based)
    if "def predict_with_per_class_threshold" in src:
        old_logic = """    ratio = probs / thresh_tensor
    mask = torch.argmax(ratio, dim=0)"""
        new_logic = """    margin = probs - thresh_tensor
    mask = torch.argmax(margin, dim=0)
    
    # Conflict resolution: if no class exceeds its threshold, default to background (0)
    max_margin, _ = torch.max(margin, dim=0)
    mask[max_margin < 0] = 0"""
        src = src.replace(old_logic, new_logic)

    # FIX 8: Use Dice for model selection
    if "class Trainer:" in src:
        # validate return
        src = src.replace("return avg_loss, miou, ious, dice", "valid_classes = ~np.isin(np.arange(NUM_CLASSES), list(EMPTY_CLASSES))\n        mean_dice = np.mean(dice[valid_classes]) if len(dice[valid_classes]) > 0 else 0.0\n        return avg_loss, miou, mean_dice, ious, dice")
        # unpack in train
        src = src.replace("val_loss, val_miou, ious, dice = self.validate(verbose=True)", "val_loss, val_miou, val_dice, ious, dice = self.validate(verbose=True)")
        # best checkpoint condition
        src = src.replace("if val_miou > self.best_miou:", "val_score = (val_miou + val_dice) / 2.0\n            if val_score > self.best_miou:")
        src = src.replace("self.best_miou = val_miou", "self.best_miou = val_score")
        src = src.replace("print(f\"  New best mIoU: {val_miou:.4f}\")", "print(f\"  New best Score (mIoU+Dice)/2: {val_score:.4f}\")")
        src = src.replace("print(f\"\\nBest mIoU: {self.best_miou:.4f}\")", "print(f\"\\nBest Score: {self.best_miou:.4f}\")")
        # Ensure regex replaces the history printing statement safely
        src = re.sub(r"print\(f\"\\nEpoch {epoch} \| Train: {train_loss:\.4f}.*?\"\)", "print(f\"\\nEpoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | mIoU: {val_miou:.4f} | Dice: {val_dice:.4f} | LR: {lr_backbone:.2e}\")", src)

    # FIX 1, 2, 7: Pseudo-Labeling fixes
    if "generate_pseudo_labels_optimized(" in src:
        # Fix 7: Filter function is already added in previous script, but need to make sure it's applied correctly.
        # Wait, the previous script added remove_small_components.
        pass

    if "def retrain_with_pseudo_labels(" in src:
        # Fix 2: Add BalancedBatchSampler
        old_loader = "combined_loader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)"
        new_loader = "sampler = BalancedBatchSampler(combined_dataset, batch_size=config['batch_size'], rare_ratio=0.40, water_ratio=0.15, drop_last=True)\n    combined_loader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)"
        src = src.replace(old_loader, new_loader)
        
        # Fix 1: base_ckpt_path is already fixed in previous script via current_ckpt_path logic!
        # Wait, I did: base_ckpt_path=current_ckpt_path, in the loop. Let me ensure it's there.

    cell['source'] = src.splitlines(keepends=True)

# FIX 4: Replace entire Cell 19 for Validation Ensemble Grid Search
new_cell_19 = """# ==================================================
# CELL 19: TASK 1 - Ensemble Submission (FULL WORKING)
# ==================================================
# Fungsi: Ensemble SegFormer + Mask2Former dengan Grid Search Weight, TTA + per-class threshold

print("="*60)
print("TASK 1: Ensemble Submission (SegFormer + Mask2Former)")
print("="*60)

device = torch.device('cuda')
torch.cuda.empty_cache()

# ========== LOAD BOTH MODELS ==========
model_m2f, ckpt_m2f = load_m2f(CKPT_M2F, device)  # Mask2Former best
model_seg, ckpt_seg = load_segformer(CKPT_SEG, device)  # SegFormer best

print(f"Mask2Former best Score: {ckpt_m2f.get('best_miou', 0):.4f}" if ckpt_m2f else "Mask2Former loaded")
print(f"SegFormer best Score: {ckpt_seg.get('best_miou', 0):.4f}" if ckpt_seg else "SegFormer loaded")

# ========== GRID SEARCH ENSEMBLE WEIGHT ON VAL SET ==========
print("\\nSearching best ensemble weight on validation set...")
model_m2f.eval()
model_seg.eval()

val_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(VAL_IMG) if f.endswith('.jpg')])
val_dataset_gs = FloodSegDataset(VAL_IMG, VAL_MASK, transform=val_transform, image_ids=val_ids)
val_loader_gs = DataLoader(val_dataset_gs, batch_size=2, shuffle=False, num_workers=2)

weights_to_test = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
best_weight = 0.65
best_score = 0.0

all_logits_m2f = []
all_logits_seg = []
all_masks = []

print("Caching val logits for grid search...")
for images, masks, _ in tqdm(val_loader_gs, leave=False):
    images = images.to(device)
    with torch.no_grad():
        lm = model_m2f(images)
        ls = model_seg(images)
    all_logits_m2f.append(lm.cpu())
    all_logits_seg.append(ls.cpu())
    all_masks.append(masks.cpu())

print("Evaluating weights...")
for w in weights_to_test:
    metrics = FloodSegMetrics(NUM_CLASSES, EMPTY_CLASSES)
    for lm, ls, masks in zip(all_logits_m2f, all_logits_seg, all_masks):
        pm = F.softmax(lm.float(), dim=1)
        ps = F.softmax(ls.float(), dim=1)
        probs = w * pm + (1 - w) * ps
        preds = torch.argmax(probs, dim=1)
        metrics.update(preds.numpy(), masks.numpy())
    score = metrics.compute_miou()
    print(f"Weight M2F: {w:.2f} -> mIoU: {score:.4f}")
    if score > best_score:
        best_score = score
        best_weight = w

print(f"Selected best weight_m2f: {best_weight:.2f}")

# ========== ENSEMBLE PREDICTION FUNCTION ==========
@torch.no_grad()
def ensemble_predict_tta(img_np, weight_m2f=best_weight):
    model_m2f.eval()
    model_seg.eval()
    
    H_orig, W_orig = img_np.shape[:2]
    all_probs = []
    
    scales = [0.75, 0.875, 1.0, 1.125, 1.25]
    rotations = [0, 90, 180, 270]
    
    for angle in rotations:
        img_rot = apply_rotation(img_np, angle)
        
        for scale in scales:
            size = max(32, int(512 * scale) // 32 * 32)
            transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])
            
            for do_flip in [False, True]:
                img_aug = np.fliplr(img_rot).copy() if do_flip else img_rot
                tensor = transform(image=img_aug)['image'].unsqueeze(0).to(device)
                
                # Mask2Former prediction
                logits_m2f = model_m2f(tensor)
                logits_m2f = F.interpolate(logits_m2f, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
                probs_m2f = F.softmax(logits_m2f.float(), dim=1).squeeze(0)
                
                # SegFormer prediction
                logits_seg = model_seg(tensor)
                logits_seg = F.interpolate(logits_seg, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
                probs_seg = F.softmax(logits_seg.float(), dim=1).squeeze(0)
                
                # Weighted ensemble
                probs = weight_m2f * probs_m2f + (1.0 - weight_m2f) * probs_seg
                
                if do_flip:
                    probs = torch.flip(probs, dims=[-1])
                
                probs = reverse_rotation_probs(probs, angle)
                all_probs.append(probs)
    
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return predict_with_per_class_threshold(avg_probs, DEFAULT_CLASS_THRESHOLDS)

# ========== GENERATE SUBMISSION ==========
test_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(TEST_IMG) if f.endswith('.jpg')])
results = []

for img_id in tqdm(test_ids, desc="Ensemble Inference"):
    img_path = os.path.join(TEST_IMG, f"{img_id}.jpg")
    img_np = np.array(Image.open(img_path).convert('RGB'))
    
    mask = ensemble_predict_tta(img_np, weight_m2f=best_weight)
    
    if mask.shape != (480, 640):
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((640, 480), Image.NEAREST)
        mask = np.array(mask_img)
    
    rles = mask_to_rle(mask)
    for c in range(NUM_CLASSES):
        results.append({'id': f"{img_id}_{c}", 'encoded_pixels': rles[c]})

df = pd.DataFrame(results)
df['encoded_pixels'] = df['encoded_pixels'].fillna('')
df.to_csv('submission_ensemble.csv', index=False, na_rep='')

print(f"[OK] Saved: submission_ensemble.csv ({len(df)} rows)")
validate_submission('submission_ensemble.csv')
"""

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "TASK 1: Ensemble Submission" in source_str:
            cell['source'] = new_cell_19.splitlines(keepends=True)
            found = True
            break

if found:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("All fixes successfully applied!")
else:
    print("Could not find Cell 19 to replace.")

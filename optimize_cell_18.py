import json

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell_18_source = """# ==================================================
# CELL 18: TASK 5 - Pseudo-Labeling (OPTIMIZED)
# ==================================================
# Fungsi: Generate pseudo-labels dari test set (Ensemble + TTA + Adaptive Threshold) dan retrain

import torch.nn.functional as F

# 1. Per-Class Adaptive Thresholds
CLASS_PSEUDO_THRESHOLDS = {
    0: 0.95,   # background
    1: 0.90,   # building flooded
    3: 0.90,   # grass
    4: 0.85,   # pool (lebih rendah)
    5: 0.90,   # road flooded
    7: 0.90,   # tree
    8: 0.75,   # vehicle (jauh lebih rendah)
    9: 0.75,   # water (jauh lebih rendah)
}

def apply_adaptive_threshold(pred_mask, max_probs, class_thresholds):
    pseudo_mask = pred_mask.clone()
    for class_id, threshold in class_thresholds.items():
        if class_id in EMPTY_CLASSES:
            continue
        low_conf = (pred_mask == class_id) & (max_probs < threshold)
        pseudo_mask[low_conf] = 255  # ignore in loss
        
    for cls_id in EMPTY_CLASSES:
        pseudo_mask[pseudo_mask == cls_id] = 0
        
    return pseudo_mask

@torch.no_grad()
def generate_pseudo_labels_optimized(model_m2f, model_seg, test_img_dir, output_dir, device, weight_m2f=0.65, scales=[0.75, 1.0, 1.25], rotations=[0]):
    \"\"\"Generate pseudo-labels with Ensemble Teacher, TTA, and Adaptive Thresholds\"\"\"
    model_m2f.eval()
    if model_seg is not None:
        model_seg.eval()
        
    test_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
    
    accepted = 0
    total_pix = 0
    conf_pix = 0
    
    for img_id in tqdm(test_ids, desc="Generating Pseudo-Labels"):
        img_path = os.path.join(test_img_dir, f"{img_id}.jpg")
        img_np = np.array(Image.open(img_path).convert('RGB'))
        H, W = img_np.shape[:2]
        
        all_probs = []
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
                    
                    # M2F
                    logits_m2f = model_m2f(tensor)
                    logits_m2f = F.interpolate(logits_m2f, size=(H, W), mode='bilinear', align_corners=False)
                    probs_m2f = F.softmax(logits_m2f.float(), dim=1).squeeze(0)
                    
                    probs = probs_m2f
                    # SegFormer if available
                    if model_seg is not None:
                        logits_seg = model_seg(tensor)
                        logits_seg = F.interpolate(logits_seg, size=(H, W), mode='bilinear', align_corners=False)
                        probs_seg = F.softmax(logits_seg.float(), dim=1).squeeze(0)
                        probs = weight_m2f * probs_m2f + (1 - weight_m2f) * probs_seg
                    
                    if do_flip:
                        probs = torch.flip(probs, dims=[-1])
                    probs = reverse_rotation_probs(probs, angle)
                    all_probs.append(probs)
                    
        avg_probs = torch.stack(all_probs).mean(dim=0)
        max_probs, pred_mask = torch.max(avg_probs, dim=0)
        
        pseudo_mask = apply_adaptive_threshold(pred_mask, max_probs, CLASS_PSEUDO_THRESHOLDS)
        
        total_pix += H * W
        conf_pix += (pseudo_mask != 255).sum().item()
        accepted += 1
        
        pseudo_np = pseudo_mask.cpu().numpy().astype(np.uint8)
        Image.fromarray(pseudo_np).save(os.path.join(output_dir, f"{img_id}.png"))
        
    print(f"Generated {accepted} pseudo-masks")
    print(f"Confident pixels (avg): {conf_pix/total_pix*100:.1f}%")
    return accepted

def retrain_with_pseudo_labels(base_ckpt_path, pseudo_mask_dir, save_path, config, class_weights, device, iteration=1):
    \"\"\"Retrain model with original + pseudo-labeled data (DENGAN RESUME)\"\"\"
    last_ckpt = os.path.join(config['save_dir'], f'last_checkpoint_pseudo_iter{iteration}.pth')
    checkpoint = None
    start_epoch = 1
    
    if os.path.exists(last_ckpt):
        print(f"[OK] Found pseudo checkpoint iter {iteration}: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location='cpu', weights_only=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"   Resume from epoch {start_epoch-1}")
    
    train_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.jpg')])
    val_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(VAL_IMG) if f.endswith('.jpg')])
    
    orig_dataset = FloodSegDataset(TRAIN_IMG, TRAIN_MASK, transform=train_transform, image_ids=train_ids)
    pseudo_dataset = PseudoLabelDataset(TEST_IMG, pseudo_mask_dir, transform=train_transform)
    val_dataset = FloodSegDataset(VAL_IMG, VAL_MASK, transform=val_transform, image_ids=val_ids)
    
    combined_dataset = torch.utils.data.ConcatDataset([orig_dataset, pseudo_dataset])
    print(f"Combined Iter {iteration}: {len(orig_dataset)} orig + {len(pseudo_dataset)} pseudo = {len(combined_dataset)}")
    
    combined_loader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader_ps = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model, _ = load_m2f(base_ckpt_path, device)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[OK] Model weights loaded from pseudo checkpoint")
        
    optimizer = configure_optimizer(model, lr_backbone=config['lr_backbone'], lr_head=config['lr_head'], weight_decay=config['weight_decay'])
    scheduler = get_poly_scheduler(optimizer, config['num_epochs'], len(combined_loader), accum_steps=config['accumulation_steps'], power=config['poly_power'], warmup_steps=config['warmup_steps'])
    
    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("[OK] Optimizer state loaded")
        except Exception as e:
            print(f"[WARN] Could not load optimizer state: {e}")
            
    loss_fn_ps = FloodSegLoss(class_weights.to(device), lovasz_weight=config['lovasz_weight'], empty_classes=EMPTY_CLASSES)
    
    trainer_ps = Trainer(
        model=model, train_loader=combined_loader, val_loader=val_loader_ps,
        loss_fn=loss_fn_ps, optimizer=optimizer, scheduler=scheduler,
        device=device, config={**config, 'save_dir': TRAIN_CONFIG['save_dir']}, 
        use_amp=True, task_name=f'pseudo_iter{iteration}'
    )
    
    if checkpoint is not None:
        trainer_ps.best_miou = checkpoint.get('best_miou', 0.0)
        trainer_ps.best_smoothed_val_loss = checkpoint.get('best_smoothed_val_loss', float('inf'))
        trainer_ps.patience_counter = checkpoint.get('patience_counter', 0)
        trainer_ps.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_miou': []})
        print(f"[OK] Trainer state restored (best_miou: {trainer_ps.best_miou:.4f})")
        
    print(f"\\n{'='*60}\\nSTARTING PSEUDO TRAINING ITERATION {iteration} FROM EPOCH {start_epoch}\\n{'='*60}\\n")
    trainer_ps.train(start_epoch=start_epoch)
    
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_miou': trainer_ps.best_miou,
        'history': trainer_ps.history,
    }, save_path)
    print(f"[OK] Saved pseudo-trained model iter {iteration} -> {save_path}")
    
    return model

# ========== RUN OPTIMIZED PSEUDO-LABELING ==========
print("="*60)
print("TASK 5: Optimized Pseudo-Labeling (Ensemble + TTA + Multi-Iter)")
print("="*60)

device = torch.device('cuda')
torch.cuda.empty_cache()

# Load initial teachers
print("Loading initial teachers...")
teacher_m2f, _ = load_m2f(CKPT_M2F, device)
teacher_seg, _ = load_segformer(CKPT_SEG, device) if 'CKPT_SEG' in globals() and os.path.exists(CKPT_SEG) else (None, None)

NUM_ITERATIONS = 3
current_teacher_m2f = teacher_m2f

for iteration in range(1, NUM_ITERATIONS + 1):
    print(f"\\n--- PSEUDO-LABELING ITERATION {iteration} ---")
    iter_pseudo_dir = f"{PSEUDO_MASK_DIR}_iter{iteration}"
    os.makedirs(iter_pseudo_dir, exist_ok=True)
    
    # Generate pseudo-labels
    generate_pseudo_labels_optimized(
        model_m2f=current_teacher_m2f,
        model_seg=teacher_seg if iteration == 1 else None, # Ensemble for first iter, then self-train
        test_img_dir=TEST_IMG,
        output_dir=iter_pseudo_dir,
        device=device,
        scales=[0.75, 1.0, 1.25], # Light TTA to keep generation reasonably fast
        rotations=[0]
    )
    
    iter_ckpt_path = CKPT_M2F_PSEUDO.replace('.pth', f'_iter{iteration}.pth')
    
    # Retrain
    current_teacher_m2f = retrain_with_pseudo_labels(
        base_ckpt_path=CKPT_M2F, # Always init from baseline to prevent drift
        pseudo_mask_dir=iter_pseudo_dir,
        save_path=iter_ckpt_path,
        config=TRAIN_CONFIG_FINETUNE,
        class_weights=CLASS_WEIGHTS_ENHANCED,
        device=device,
        iteration=iteration
    )
    
    # Update global variable for subsequent tasks
    CKPT_M2F_PSEUDO = iter_ckpt_path

print("[OK] Task 5 (Optimized Pseudo-labeling) completed!")
"""

# Format the string into a list of lines with newlines
new_cell_lines = new_cell_18_source.splitlines(keepends=True)

# Find Cell 18
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "TASK 5 - Pseudo-Labeling" in source_str:
            cell['source'] = new_cell_lines
            found = True
            break

if found:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Cell 18 successfully replaced with the optimized version.")
else:
    print("Could not find Cell 18 (TASK 5) in the notebook.")

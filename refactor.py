import json
import re

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
        
    src = "".join(cell['source'])
    
    # BATCH A: Emojis
    src = src.replace('✅', '[OK]')
    src = src.replace('⚠️', '[WARN]')
    src = src.replace('❌', '[FAIL]')
    src = src.replace('→', '->')
    
    # CRIT-1: Forward pass log probs
    if "class Mask2FormerFlood(nn.Module):" in src:
        src = src.replace("return seg_maps", "return torch.log(seg_maps.clamp(min=1e-8))")
        
    # CRIT-2: Sampler
    if "class BalancedBatchSampler(BatchSampler):" in src:
        old_iter = """    def __iter__(self):
        n_water = max(1, int(self.batch_size * self.water_ratio))
        n_rare = max(1, int(self.batch_size * self.rare_ratio)) - n_water
        n_normal = self.batch_size - n_water - n_rare

        water_pool = self.water_indices.copy()
        rare_pool = [i for i in self.rare_indices if i not in self.water_set]
        normal_pool = self.normal_indices.copy()

        random.shuffle(water_pool)
        random.shuffle(rare_pool)
        random.shuffle(normal_pool)

        i_water, i_rare, i_normal = 0, 0, 0
        n_batches = len(self)

        for _ in range(n_batches):
            batch = []
            for _ in range(n_water):
                if i_water >= len(water_pool):
                    random.shuffle(water_pool)
                    i_water = 0
                batch.append(water_pool[i_water])
                i_water += 1
            for _ in range(n_rare):
                if i_rare >= len(rare_pool):
                    random.shuffle(rare_pool)
                    i_rare = 0
                batch.append(rare_pool[i_rare])
                i_rare += 1
            for _ in range(n_normal):
                if i_normal >= len(normal_pool):
                    random.shuffle(normal_pool)
                    i_normal = 0
                batch.append(normal_pool[i_normal])
                i_normal += 1
            random.shuffle(batch)
            yield batch"""
            
        new_iter = """    def __iter__(self):
        water_pool = self.water_indices.copy()
        rare_pool = [i for i in self.rare_indices if i not in self.water_set]
        normal_pool = self.normal_indices.copy()

        random.shuffle(water_pool)
        random.shuffle(rare_pool)
        random.shuffle(normal_pool)

        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            for _ in range(self.batch_size):
                r = random.random()
                if r < self.water_ratio:
                    if not water_pool:
                        water_pool = self.water_indices.copy()
                        random.shuffle(water_pool)
                    batch.append(water_pool.pop())
                elif r < self.rare_ratio:
                    if not rare_pool:
                        rare_pool = [i for i in self.rare_indices if i not in self.water_set]
                        random.shuffle(rare_pool)
                    batch.append(rare_pool.pop())
                else:
                    if not normal_pool:
                        normal_pool = self.normal_indices.copy()
                        random.shuffle(normal_pool)
                    batch.append(normal_pool.pop())
            yield batch"""
        src = src.replace(old_iter, new_iter)
        
    # CRIT-5: Per-class threshold logic
    if "def predict_with_per_class_threshold" in src:
        old_thresh = """    # Rare classes diproses dulu dengan threshold rendah
    priority_order = [8, 9, 4, 1, 5, 7, 3, 0]
    for cls_id in priority_order:
        if cls_id in EMPTY_CLASSES:
            continue
        threshold = class_thresholds.get(cls_id, 0.5)
        cls_mask = (probs[cls_id] > threshold) & (mask == -1)
        mask[cls_mask] = cls_id
    
    # Sisa pixel -> argmax
    remaining = (mask == -1)
    if remaining.sum() > 0:
        mask[remaining] = torch.argmax(probs, dim=0)[remaining]"""
        
        new_thresh = """    # Gunakan ratio probabilitas terhadap threshold
    thresh_tensor = torch.ones((C, H, W), device=device) * 0.5
    for c, t in class_thresholds.items():
        thresh_tensor[c] = t
        
    ratio = probs / thresh_tensor
    mask = torch.argmax(ratio, dim=0)"""
        src = src.replace(old_thresh, new_thresh)

    # CRIT-3 & BATCH B: Trainer checkpoints
    if "class Trainer:" in src:
        src = src.replace("def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, config, use_amp=True):",
                          "def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, config, use_amp=True, task_name='baseline'):")
        src = src.replace("self.last_checkpoint_path = os.path.join(config['save_dir'], 'last_checkpoint.pth')",
                          "self.last_checkpoint_path = os.path.join(config['save_dir'], f'last_checkpoint_{task_name}.pth')\n        self.best_checkpoint_path = os.path.join(config['save_dir'], f'best_miou_{task_name}.pth')\n        self.best_loss_path = os.path.join(config['save_dir'], f'best_loss_{task_name}.pth')")
        
        src = src.replace("torch.save(checkpoint, os.path.join(self.config['save_dir'], 'best_miou.pth'))",
                          "torch.save(checkpoint, self.best_checkpoint_path)")
        src = src.replace("torch.save(checkpoint, os.path.join(self.config['save_dir'], 'best_loss.pth'))",
                          "torch.save(checkpoint, self.best_loss_path)")

    # Trainer instantiation updates
    if "config=TRAIN_CONFIG," in src:
        src = re.sub(r"use_amp=True\s*\)", "use_amp=True,\n    task_name='baseline'\n)", src)
    if "config={**config" in src:
        src = re.sub(r"use_amp=True\s*\)", "use_amp=True,\n        task_name='pseudo'\n    )", src)
    if "config=TRAIN_CONFIG_FINETUNE" in src:
        src = re.sub(r"use_amp=True\s*\)", "use_amp=True,\n    task_name='finetune'\n)", src)
    if "config={'batch_size': 2" in src:
        src = re.sub(r"use_amp=True\s*\)", "use_amp=True,\n    task_name='weighted'\n)", src)

    # Dead code removal
    if "loss_fn = FloodSegLoss(CLASS_WEIGHTS," in src:
        src = src.replace("loss_fn = FloodSegLoss(CLASS_WEIGHTS, lovasz_weight=TRAIN_CONFIG['lovasz_weight'], empty_classes=EMPTY_CLASSES)\n", "")

    # BATCH E: np.rot90 optimization
    if "def apply_rotation(" in src:
        old_apply = """def apply_rotation(img_np, angle):
    \"\"\"Rotate image by angle degrees (clockwise)\"\"\"
    if angle == 0:
        return img_np
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)"""
        
        new_apply = """def apply_rotation(img_np, angle):
    \"\"\"Rotate image by angle degrees (clockwise)\"\"\"
    if angle == 0:
        return img_np
    elif angle == 90:
        return np.rot90(img_np, k=-1)
    elif angle == 180:
        return np.rot90(img_np, k=2)
    elif angle == 270:
        return np.rot90(img_np, k=1)
    
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)"""
        src = src.replace(old_apply, new_apply)
        
    if "def reverse_rotation_probs(" in src:
        old_rev = """def reverse_rotation_probs(probs_tensor, angle):
    \"\"\"Reverse rotation on probability map\"\"\"
    if angle == 0:
        return probs_tensor
    C, H, W = probs_tensor.shape
    arr = probs_tensor.permute(1, 2, 0).cpu().numpy()
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = np.stack([
        cv2.warpAffine(arr[:, :, c], M, (W, H), borderMode=cv2.BORDER_REFLECT)
        for c in range(C)
    ], axis=0)
    return torch.from_numpy(rotated).to(probs_tensor.device)"""
        
        new_rev = """def reverse_rotation_probs(probs_tensor, angle):
    \"\"\"Reverse rotation on probability map\"\"\"
    if angle == 0:
        return probs_tensor
        
    k_map = {90: 1, 180: 2, 270: -1}
    if angle in k_map:
        return torch.rot90(probs_tensor, k=k_map[angle], dims=[1, 2])
        
    C, H, W = probs_tensor.shape
    arr = probs_tensor.permute(1, 2, 0).cpu().numpy()
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = np.stack([
        cv2.warpAffine(arr[:, :, c], M, (W, H), borderMode=cv2.BORDER_REFLECT)
        for c in range(C)
    ], axis=0)
    return torch.from_numpy(rotated).to(probs_tensor.device)"""
        src = src.replace(old_rev, new_rev)

    cell['source'] = src.splitlines(keepends=True)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Notebook updated successfully.")

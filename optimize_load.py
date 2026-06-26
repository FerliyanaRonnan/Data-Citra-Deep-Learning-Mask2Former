import json

file_path = "Mask2Form-Pseudo.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        if "def retrain_with_pseudo_labels(" in source_str:
            old_logic = """    if os.path.exists(last_ckpt):
        print(f"[OK] Found pseudo checkpoint iter {iteration}: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location='cpu', weights_only=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"   Resume from epoch {start_epoch-1}")
    
    train_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.jpg')])
    val_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(VAL_IMG) if f.endswith('.jpg')])
    
    orig_dataset = FloodSegDataset(TRAIN_IMG, TRAIN_MASK, transform=train_transform, image_ids=train_ids)
    pseudo_dataset = PseudoLabelDataset(TEST_IMG, pseudo_mask_dir, transform=train_transform)
    val_dataset = FloodSegDataset(VAL_IMG, VAL_MASK, transform=val_transform, image_ids=val_ids)
    
    # Fix 2: Wrap the datasets to support BalancedBatchSampler
    combined_dataset = CombinedDatasetWrapper(orig_dataset, pseudo_dataset)
    print(f"Combined Iter {iteration}: {len(orig_dataset)} orig + {len(pseudo_dataset)} pseudo = {len(combined_dataset)}")
    
    sampler = BalancedBatchSampler(combined_dataset, batch_size=config['batch_size'], rare_ratio=0.40, water_ratio=0.15, drop_last=True)
    combined_loader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)
    
    val_loader_ps = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model, _ = load_m2f(base_ckpt_path, device)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("[OK] Model weights loaded from pseudo checkpoint")"""
            
            new_logic = """    if os.path.exists(last_ckpt):
        print(f"[OK] Found pseudo checkpoint iter {iteration}: {last_ckpt}")
        model, checkpoint = load_m2f(last_ckpt, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"   Resume from epoch {start_epoch-1}")
    else:
        model, _ = load_m2f(base_ckpt_path, device)
        
    train_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(TRAIN_IMG) if f.endswith('.jpg')])
    val_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(VAL_IMG) if f.endswith('.jpg')])
    
    orig_dataset = FloodSegDataset(TRAIN_IMG, TRAIN_MASK, transform=train_transform, image_ids=train_ids)
    pseudo_dataset = PseudoLabelDataset(TEST_IMG, pseudo_mask_dir, transform=train_transform)
    val_dataset = FloodSegDataset(VAL_IMG, VAL_MASK, transform=val_transform, image_ids=val_ids)
    
    # Fix 2: Wrap the datasets to support BalancedBatchSampler
    combined_dataset = CombinedDatasetWrapper(orig_dataset, pseudo_dataset)
    print(f"Combined Iter {iteration}: {len(orig_dataset)} orig + {len(pseudo_dataset)} pseudo = {len(combined_dataset)}")
    
    sampler = BalancedBatchSampler(combined_dataset, batch_size=config['batch_size'], rare_ratio=0.40, water_ratio=0.15, drop_last=True)
    combined_loader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)
    
    val_loader_ps = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)"""
            
            source_str = source_str.replace(old_logic, new_logic)
            cell['source'] = source_str.splitlines(keepends=True)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Minor optimization applied: Avoided double-loading model weights during resume.")

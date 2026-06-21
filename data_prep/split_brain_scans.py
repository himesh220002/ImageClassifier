import os
import shutil
import random

def split_brain_scans(src_dir="data/Brain_scan", dest_dir="data/raw/brain_scans", split_ratio=0.8):
    os.makedirs(dest_dir, exist_ok=True)
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for cls in classes:
        cls_src = os.path.join(src_dir, cls)
        cls_train = os.path.join(train_dir, cls)
        cls_val = os.path.join(val_dir, cls)
        
        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_val, exist_ok=True)
        
        images = [f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        for img in train_imgs:
            shutil.copy2(os.path.join(cls_src, img), os.path.join(cls_train, img))
            
        for img in val_imgs:
            shutil.copy2(os.path.join(cls_src, img), os.path.join(cls_val, img))
            
        print(f"Processed {cls}: {len(train_imgs)} train, {len(val_imgs)} val.")
        
if __name__ == "__main__":
    random.seed(42)
    split_brain_scans()

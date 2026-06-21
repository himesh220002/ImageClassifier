import os
import shutil
import random
import pandas as pd

def split_ecommerce(csv_path="data/e-commerce_products/fashion.csv", 
                    img_base_dir="data/e-commerce_products", 
                    dest_dir="data/raw/ecommerce", 
                    split_ratio=0.8,
                    target_col="SubCategory"):
    
    os.makedirs(dest_dir, exist_ok=True)
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # We need to find the images. In fashion.csv, they have ProductId.jpg
    # The actual images are in Apparel/Boys/Images/images_with_product_ids/...
    # Let's do a recursive search to build a fast lookup map of ProductId.jpg -> full_path
    img_map = {}
    print("Indexing images...")
    for root, _, files in os.walk(img_base_dir):
        if "images_with_product_ids" in root:
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_map[file] = os.path.join(root, file)
                    
    print(f"Found {len(img_map)} images on disk.")
    
    # Filter df to only rows where we have the image
    df = df[df['Image'].isin(img_map)]
    print(f"Matched {len(df)} images with CSV records.")
    
    # Group by target class
    classes = df[target_col].unique()
    
    for cls in classes:
        # Avoid creating directories for very rare classes (e.g. less than 10 images)
        cls_df = df[df[target_col] == cls]
        if len(cls_df) < 10:
            continue
            
        # Create safe directory name
        safe_cls = str(cls).replace("/", "_").replace(" ", "_")
        cls_train = os.path.join(train_dir, safe_cls)
        cls_val = os.path.join(val_dir, safe_cls)
        
        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_val, exist_ok=True)
        
        images = cls_df['Image'].tolist()
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        for img in train_imgs:
            shutil.copy2(img_map[img], os.path.join(cls_train, img))
            
        for img in val_imgs:
            shutil.copy2(img_map[img], os.path.join(cls_val, img))
            
        print(f"Processed {safe_cls}: {len(train_imgs)} train, {len(val_imgs)} val.")

if __name__ == "__main__":
    random.seed(42)
    split_ecommerce()

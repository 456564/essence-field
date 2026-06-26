"""从 Caltech101 每个类别随机取一张图"""
import os, shutil, random

random.seed(42)
src = 'data/caltech101/101_ObjectCategories'
dst = 'test_output/real_images'
os.makedirs(dst, exist_ok=True)

for cat in sorted(os.listdir(src)):
    cat_dir = os.path.join(src, cat)
    if os.path.isdir(cat_dir):
        imgs = [f for f in os.listdir(cat_dir) if f.endswith(('.jpg','.png','.jpeg'))]
        if imgs:
            img = random.choice(imgs)
            ext = os.path.splitext(img)[1]
            shutil.copy(os.path.join(cat_dir, img), os.path.join(dst, f'{cat}{ext}'))
            print(f'{cat}: {img}')

print(f'\nDone - {len(os.listdir(dst))} images copied to {dst}')

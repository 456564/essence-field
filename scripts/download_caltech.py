"""下载 Caltech101 并提取每类一张测试图"""
import torchvision, os

os.makedirs('data/caltech101', exist_ok=True)
os.makedirs('test_output/real_images', exist_ok=True)

print('Downloading Caltech101...')
ds = torchvision.datasets.Caltech101('data/caltech101', download=True)
print(f'Total: {len(ds)} images')

seen = set()
count = 0
for img, label in ds:
    cat_name = ds.categories[label]
    if cat_name not in seen:
        seen.add(cat_name)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(f'test_output/real_images/caltech_{count:03d}_{cat_name}.png')
        count += 1

print(f'Saved {count} images from {len(ds.categories)} categories')

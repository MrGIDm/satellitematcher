"""
Простая подготовка датасета для задания
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Пути к данным
ONERA_DIR = Path(r"D:\PythonProject\data\onera-dataset")
OUTPUT_DIR = Path("dataset_pairs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("🛰️ Подготовка датасета для Satellite Image Matching")
print("="*60)

# Собираем пары из ONERA
pairs = []
cities = [d for d in ONERA_DIR.iterdir() if d.is_dir()]

print(f"\n✓ Найдено городов: {len(cities)}")

for city_dir in tqdm(cities, desc="Обработка городов"):
    img1_path = city_dir / "pair" / "img1.png"
    img2_path = city_dir / "pair" / "img2.png"

    if img1_path.exists() and img2_path.exists():
        pairs.append({
            'city': city_dir.name,
            'img1': str(img1_path),
            'img2': str(img2_path)
        })

print(f"✓ Найдено пар изображений: {len(pairs)}")

# Сохраняем информацию о датасете
dataset_info = {
    'num_pairs': len(pairs),
    'pairs': pairs
}

with open(OUTPUT_DIR / 'dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2)

print(f"\n✓ Датасет готов!")
print(f"✓ Сохранено в: {OUTPUT_DIR / 'dataset_info.json'}")
print(f"✓ Всего пар: {len(pairs)}")
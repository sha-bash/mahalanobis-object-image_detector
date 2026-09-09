import csv
import json
import os

csv_file = r"C:\Users\sha-b\Downloads\DAY_DASIAMRPN_GAZEL.csv"
# Используем только кадры, для которых есть разметка в CSV
frames_with_gt = set()
data = {}
with open(csv_file, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        frame_idx = int(float(row['frame']))
        frames_with_gt.add(frame_idx)
        data[frame_idx] = {
            'x': float(row['x']),
            'y': float(row['y']),
            'w': float(row['w']),
            'h': float(row['h'])
        }

# Создаём COCO только для этих кадров
coco = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'van'}]}
ann_id = 1
for frame_idx in sorted(frames_with_gt):
    image_id = frame_idx  # можно использовать сам номер кадра как id
    file_name = f"frame_{frame_idx:08d}.png"   # <-- правильное имя
    coco['images'].append({
        'id': image_id,
        'file_name': file_name,
        'width': 3840,
        'height': 2160
    })
    bbox = [data[frame_idx]['x'], data[frame_idx]['y'], data[frame_idx]['w'], data[frame_idx]['h']]
    area = bbox[2] * bbox[3]
    coco['annotations'].append({
        'id': ann_id,
        'image_id': image_id,
        'category_id': 1,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0
    })
    ann_id += 1

os.makedirs('annotations', exist_ok=True)
with open('annotations/instances.json', 'w') as f:
    json.dump(coco, f, indent=2)
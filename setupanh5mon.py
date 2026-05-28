import os
import shutil

# ĐƯỜNG DẪN - sửa lại cho đúng với máy bạn
source_root = "Vietnamese_StreetFood_14Class"   # Nếu có thư mục con, thêm vào, ví dụ "streetfood_full/Vietnamese_StreetFood_14Class"
target_root = "data/dishes"

# Mapping ID -> tên món (dựa trên dataset Vietnamese_StreetFood)
# ID tham khảo: 11: beef pho, 2: grilled pork banh mi, 3: hue-style spicy beef noodle soup, 9: broken rice, 12: spring rolls
id_to_name = {
    11: "beef pho",
    2: "grilled pork banh mi",
    3: "hue-style spicy beef noodle soup",
    9: "broken rice",
    12: "spring rolls"
}

desired_dishes = set(id_to_name.values())
name_mapping = {
    "beef pho": "pho",
    "grilled pork banh mi": "banh_mi",
    "hue-style spicy beef noodle soup": "bun_bo_hue",
    "broken rice": "com_tam",
    "spring rolls": "goi_cuon"
}

# Tạo thư mục đích
for split in ['train', 'test']:
    for dish in name_mapping.values():
        os.makedirs(os.path.join(target_root, split, dish), exist_ok=True)

# Duyệt qua các split: train, valid, test
for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(source_root, split, 'images')
    label_dir = os.path.join(source_root, split, 'labels')
    if not os.path.exists(img_dir):
        print(f"⚠️ Bỏ qua {split}: không tìm thấy {img_dir}")
        continue
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, label_file), 'r') as f:
            line = f.readline().strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            dish_name = id_to_name.get(class_id, "")
        if dish_name not in desired_dishes:
            continue
        # Tìm file ảnh
        img_name = label_file.replace('.txt', '.jpg')
        if not os.path.exists(os.path.join(img_dir, img_name)):
            img_name = label_file.replace('.txt', '.png')
        src_img = os.path.join(img_dir, img_name)
        if not os.path.exists(src_img):
            continue
        target_subfolder = name_mapping[dish_name]
        if split in ['train', 'valid']:
            dest_folder = os.path.join(target_root, 'train', target_subfolder)
        else:
            dest_folder = os.path.join(target_root, 'test', target_subfolder)
        unique_name = f"{split}_{img_name}"
        shutil.copy(src_img, os.path.join(dest_folder, unique_name))

print("✅ Đã copy xong ảnh cho 5 món vào data/dishes")
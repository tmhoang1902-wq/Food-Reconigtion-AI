"""
Bài 2 - Vietnamese Dishes Recognition (ít nhất 5 món)
Nhận diện các món ăn Việt Nam bằng CNN
Các món: Phở, Bánh Mì, Bún Bò Huế, Cơm Tấm, Bánh Xèo, Gỏi Cuốn
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras          # 👈 THÊM DÒNG NÀY
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
IMG_SIZE    = (150, 150)
BATCH_SIZE  = 32
EPOCHS      = 60
NUM_CLASSES = 5           # ít nhất 5 món, ở đây dùng 6

# Tên các lớp (món ăn)
CLASS_NAMES = ['pho', 'banh_mi', 'bun_bo_hue', 'com_tam', 'goi_cuon']
CLASS_DISPLAY = {
    'pho'        : 'Phở',
    'banh_mi'    : 'Bánh Mì',
    'bun_bo_hue' : 'Bún Bò Huế',
    'com_tam'    : 'Cơm Tấm',
    'goi_cuon'   : 'Gỏi Cuốn',
}

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU
# Cấu trúc thư mục:
#   data/dishes/
#       train/
#           pho/
#           banh_mi/
#           bun_bo_hue/
#           com_tam/
#           banh_xeo/
#           goi_cuon/
#       test/
#           (tương tự)
# ============================================================
TRAIN_DIR = 'data/dishes/train'
TEST_DIR  = 'data/dishes/test'


# ============================================================
# BƯỚC 1: LOAD & AUGMENT DỮ LIỆU
# ============================================================
def load_data():
    # Augmentation mạnh hơn cho ảnh đồ ăn (góc chụp đa dạng, ánh sáng thay đổi)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,  # thay đổi màu sắc nhẹ
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("Mapping lớp:", train_gen.class_indices)
    return train_gen, val_gen, test_gen


# ============================================================
# BƯỚC 2: XÂY DỰNG MÔ HÌNH CNN (dùng Transfer Learning MobileNetV2)
# Dùng MobileNetV2 vì ảnh đồ ăn cần feature phức tạp (màu sắc, kết cấu)
# ============================================================
def build_model_transfer():
    # Load MobileNetV2 đã pre-trained trên ImageNet, bỏ lớp top
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Đóng băng base model ban đầu
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model, base_model


# ============================================================
# (Tuỳ chọn) Mô hình CNN thuần nếu không muốn dùng Transfer Learning
# ============================================================
def build_model_cnn():
    model = Sequential([
        Conv2D(32,  (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64,  (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


# ============================================================
# BƯỚC 3: HUẤN LUYỆN (2 giai đoạn với Transfer Learning)
# ============================================================
def train_model(model, base_model, train_gen, val_gen):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('dishes_best_model.keras', save_best_only=True, monitor='val_accuracy', save_format='h5'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print("\n=== GIAI ĐOẠN 1: Huấn luyện lớp top (base model đóng băng) ===")
    history1 = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Giai đoạn 2: Mở băng một phần base model để fine-tune
    print("\n=== GIAI ĐOẠN 2: Fine-tuning (mở 30 lớp cuối base model) ===")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lr nhỏ hơn
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Ghép history 2 giai đoạn
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    return combined_history


# ============================================================
# BƯỚC 4: VẼ ĐỒ THỊ
# ============================================================
def plot_history(history_dict):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'],     label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title('Loss theo Epoch - Nhận Diện Món Ăn Việt')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'],     label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy theo Epoch - Nhận Diện Món Ăn Việt')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.tight_layout()
    plt.savefig('dishes_training_history.png', dpi=150)
    plt.show()


# ============================================================
# BƯỚC 5: ĐÁNH GIÁ VÀ DỰ ĐOÁN
# ============================================================
def evaluate_and_predict(model, test_gen):
    loss, accuracy = model.evaluate(test_gen)
    print(f"\n===== KẾT QUẢ TRÊN TẬP TEST =====")
    print(f"Loss    : {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\n--- Dự đoán ảnh mẫu ---")
    sample_path = 'sample_dish.jpg'   # <-- thay bằng đường dẫn ảnh thực
    if os.path.exists(sample_path):
        img = tf.keras.preprocessing.image.load_img(sample_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction    = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        confidence    = prediction[0][predicted_idx] * 100
        predicted_key = CLASS_NAMES[predicted_idx]

        print(f"Món ăn dự đoán : {CLASS_DISPLAY.get(predicted_key, predicted_key)}")
        print(f"Độ tin cậy     : {confidence:.2f}%")

        print("\nXác suất từng món:")
        for key, prob in zip(CLASS_NAMES, prediction[0]):
            bar = '█' * int(prob * 30)
            display = CLASS_DISPLAY.get(key, key)
            print(f"  {display:<15}: {bar:<30} {prob*100:.1f}%")
    else:
        print(f"(Không tìm thấy '{sample_path}', bỏ qua dự đoán mẫu)")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU khả dụng: {gpus if gpus else 'Không có GPU, dùng CPU'}")

    USE_TRANSFER_LEARNING = True   # Đặt False để dùng CNN thuần

    if not os.path.exists(TRAIN_DIR):
        print(f"[LỖI] Không tìm thấy: {TRAIN_DIR}")
        print("Vui lòng tạo cấu trúc thư mục như hướng dẫn trên đầu file.")
    else:
        train_gen, val_gen, test_gen = load_data()

        if USE_TRANSFER_LEARNING:
            model, base_model = build_model_transfer()
            history = train_model(model, base_model, train_gen, val_gen)
        else:
            model   = build_model_cnn()
            base_model = None
            from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint('dishes_best_model.keras', save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ]
            h = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)
            history = h.history

        plot_history(history)
        evaluate_and_predict(model, test_gen)
        model.save('dishes_final_model.h5', save_format='h5')
        print("\nĐã lưu mô hình: dishes_final_model.h5")

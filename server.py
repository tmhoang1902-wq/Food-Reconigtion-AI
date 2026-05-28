import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
 
# ============================================================
# CẤU HÌNH — điều chỉnh nếu cần
# ============================================================
MODEL_PATH = 'dishes_best_model.keras'   # hoặc 'dishes_final_model.keras'
IMG_SIZE   = (150, 150)
PORT       = 5001
 
CLASS_NAMES = ['banh_mi', 'bun_bo_hue', 'com_tam', 'goi_cuon', 'pho']
CLASS_DISPLAY = {
    'pho'        : 'Phở',
    'banh_mi'    : 'Bánh Mì',
    'bun_bo_hue' : 'Bún Bò Huế',
    'com_tam'    : 'Cơm Tấm',
    'goi_cuon'   : 'Gỏi Cuốn',
}
 
# ============================================================
# KHỞI ĐỘNG
# ============================================================
app = Flask(__name__)
CORS(app)   # Cho phép trang HTML gọi API từ file:// hoặc domain khác
 
# Load model một lần duy nhất khi khởi động
print(f"Đang tải mô hình: {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy '{MODEL_PATH}'.\n"
        "Hãy chạy bai2_vietnamese_dishes_recognition.py để train và lưu mô hình trước."
    )
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Tải mô hình thành công!")
 
 
# ============================================================
# HELPER
# ============================================================
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Đọc bytes ảnh, resize và chuẩn hoá về [0,1]."""
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
 
 
# ============================================================
# ENDPOINT CHÍNH
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra có file ảnh gửi lên không
    if 'image' not in request.files:
        return jsonify({'error': 'Thiếu field "image" trong request'}), 400
 
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'File rỗng'}), 400
 
    try:
        img_array = preprocess_image(file.read())
    except Exception as e:
        return jsonify({'error': f'Không đọc được ảnh: {str(e)}'}), 400
 
    # Dự đoán
    preds = model.predict(img_array, verbose=0)[0]   # shape: (NUM_CLASSES,)
    predicted_idx = int(np.argmax(preds))
    predicted_key = CLASS_NAMES[predicted_idx]
    confidence    = float(preds[predicted_idx])
 
    all_probs = {key: float(preds[i]) for i, key in enumerate(CLASS_NAMES)}
 
    return jsonify({
        'dish'      : predicted_key,
        'display'   : CLASS_DISPLAY.get(predicted_key, predicted_key),
        'confidence': confidence,
        'all'       : all_probs
    })
 
 
@app.route('/health', methods=['GET'])
def health():
    """Kiểm tra server còn sống không."""
    return jsonify({'status': 'ok', 'model': MODEL_PATH})
 
 
# ============================================================
# CHẠY SERVER
# ============================================================
if __name__ == '__main__':
    print(f"\n🚀 Server đang chạy tại http://localhost:{PORT}")
    print("   Nhấn Ctrl+C để dừng\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)
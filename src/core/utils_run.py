import copy # Import thư viện copy để sao chép dữ liệu (đặc biệt là dictionary)
from src.models.cnn_trainer import CNNTrainer # Import lớp Controller model (quản lý CNN/SE-CNN)
from src.models.resnet_trainer import ResNetTrainer # Import lớp Controller cho ResNet (dù không dùng trực tiếp trong file này)
from src.core.tools import load_data_catsVsdogs # Import hàm load dữ liệu

# Hàm chạy toàn bộ quy trình: Khởi tạo -> Huấn luyện -> Đánh giá -> Test cho 1 kiến trúc mạng
def run_for_arch(
    params, # Dictionary chứa cấu hình
    arch_name, # Tên kiến trúc (vd: 'cnn', 'se')
    train, # Tập dữ liệu huấn luyện (có thể là None)
    val, # Tập dữ liệu xác thực (có thể là None)
    force_train=False, # Cờ ép buộc phải train lại từ đầu (bỏ qua checkpoint)
    test_folder_path=None, # Đường dẫn tới thư mục test
    stop_requested=None, # Cờ dừng khẩn cấp (thường dùng cho UI)
    progress_callback=None, # Hàm callback cập nhật tiến trình (dùng cho UI)
):
    p = copy.deepcopy(params) # Sao chép sâu (deep copy) params để không làm thay đổi biến gốc
    
    # Thiết lập tên kiến trúc
    use_se = False
    if arch_name in ('se', 'se_cnn'):
        p['architecture'] = 'se' # Dùng SE-CNN
        use_se = True
    else:
        p['architecture'] = 'cnn' # Dùng CNN thường (MLP)

    # Đảm bảo mỗi mô hình sẽ được lưu vào một thư mục riêng biệt để không ghi đè lên nhau
    base_savedir = str(p.get('savedir', 'CNN_catsVsdogs')).replace('\\', '/').rstrip('/')
    if base_savedir.startswith('checkpoints/'):
        base_savedir = base_savedir[12:]
    p['savedir'] = f"{base_savedir}/{p['architecture']}"
    p['savename'] = p.get('savename', 'model.pth') # Tên file lưu trọng số

    m = CNNTrainer(p, use_se=use_se) # Khởi tạo Controller mô hình với params đã chỉnh sửa

    # Khối logic huấn luyện
    if force_train or p.get('rebuild', False):
        train_data, val_data = train, val
        # Nếu chưa truyền data vào, tự động load data từ thư mục
        if train_data is None or val_data is None:
            train_data, val_data = load_data_catsVsdogs(p)

        # Bắt đầu quá trình hội tụ (Train + Validation + Early Stopping)
        m.converge(
            train_data,
            val_data,
            stop_requested=stop_requested,
            progress_callback=progress_callback,
        )
        print(f"[INFO] Training for {p['architecture']} completed with strict checkpoint flow.")
    else:
        # Nếu không ép train, chỉ cần load mô hình từ checkpoint có sẵn
        print(f"[INFO] Loading model for architecture: {p['architecture']}")
        m.load_model(load_optimizer=False) # Load model (không cần load optimizer vì không train tiếp)
        print(f"[INFO] Load completed for architecture: {p['architecture']}")

    val_loss = None
    val_acc = None
    # Đảm bảo có tập validation để đánh giá
    if val is None:
        _, val = load_data_catsVsdogs(p)
    # Chạy hàm đánh giá độ chính xác của mô hình
    val_loss, val_acc = m.evaluate(val)
    print(f"Validation loss for {p['architecture']}: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # Nếu người dùng truyền vào 1 thư mục chứa ảnh test lẻ
    results = None
    if test_folder_path is not None:
        results = m.test_folder(test_folder_path) # Dự đoán và sinh Grad-CAM cho từng ảnh

    return val_loss, m, results # Trả về Loss, object mô hình và kết quả test

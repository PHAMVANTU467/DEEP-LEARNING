import torch # Import PyTorch
from src.core.tools import * # Import tất cả hàm và công cụ xử lý dữ liệu
from src.models.cnn_trainer import * # Import Controller của CNN/SE-CNN
from src.models.resnet_trainer import ResNetTrainer # Import Controller của ResNet

# Khởi tạo một từ điển để chứa toàn bộ siêu tham số (Hyperparameters) của mô hình
parameters = {}

# Đường dẫn thư mục ảnh
parameters['path_cats'] = 'PetImages/Cat'
parameters['path_dogs'] = 'PetImages/Dog'
# Kích thước ảnh sau khi resize (50x50 pixel)
parameters['size'] = 50

# Số kênh của ảnh đầu vào (ảnh màu RGB = 3)
parameters['input_channel'] = 3

# Cấu trúc của các lớp Tích chập (Convolution Layers)
parameters['conv_output_channels'] = [32,64,128] # Số bộ lọc ở từng lớp
parameters['conv_kernel_size'] = [5,5,5] # Kích thước cửa sổ trượt (5x5)
parameters['stride_size'] = [1,1,1] # Bước nhảy của cửa sổ trượt
parameters['padding_size'] = [0,0,0] # Thêm viền số 0 (không thêm)

# Cấu trúc của các lớp Gộp (Pooling Layers)
parameters['pool_kernel_size'] = [2,2,2] # Kích thước cửa sổ gộp (2x2)
parameters['pool_stride_size'] = [2,2,2] # Bước nhảy
parameters['pool_padding_size'] = [0,0,0]

# Chọn thiết bị chạy (nếu có GPU NVIDIA thì dùng cuda, không thì cpu)
parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tính toán tự động kích thước vector sau khi đi qua các lớp Conv+Pool
parameters['output_dimen_cnn'] = size_conv_output(parameters)

# Số neural lớp ẩn đầu vào của Fully Connected Layer
parameters['# inputs'] = parameters['output_dimen_cnn']
# Số neural đầu ra (bằng 2 vì có 2 class: Chó và Mèo)
parameters['# outputs'] = 2

# Tỉ lệ chia tập dữ liệu
parameters['validation'] = 0.1 # 10% dùng để kiểm tra trong lúc huấn luyện
parameters['training'] = 0.9 # 90% dùng để huấn luyện

# Kích thước batch (lô dữ liệu đưa vào mỗi lần cập nhật trọng số)
parameters['batch_size_training'] = 64
parameters['batch_size_validation'] = 64

# Tốc độ học (learning rate)
parameters['learning rate'] = 1e-3
# Số vòng lặp huấn luyện ban đầu (có thể tăng thêm lúc chạy)
parameters['epochs'] = 5

# Thư mục lưu trọng số
parameters['savedir'] = 'CNN_catsVsdogs'
# Tên file trọng số tốt nhất (checkpoint)
parameters['savename'] = 'best_model.pth'

# Tên file đóng gói toàn bộ dataset
parameters['dataset'] = 'dataset.npy'

import copy # Dùng để sao chép từ điển
import webbrowser # Dùng để mở trình duyệt
import urllib.request # Dùng để mã hoá URL
import json
import pathlib # Dùng để thao tác đường dẫn
import os # Dùng thao tác thư mục/hệ điều hành

# Hàm bọc quy trình huấn luyện cho 1 kiến trúc
def run_for_arch(params, arch_name, train, val, force_train=False, test_folder_path=None):
    p = copy.deepcopy(params) # Sao chép params để không làm ảnh hưởng params gốc
    # Đặt tên kiến trúc để file model.py biết cần khởi tạo mạng nào
    if arch_name in ('se', 'se_cnn'):
        p['architecture'] = 'se'
    else:
        p['architecture'] = 'cnn'

    # Tạo thư mục lưu trữ độc lập cho kiến trúc này
    p['savedir'] = f"{p.get('savedir','models')}_{p['architecture']}"
    p['savename'] = p.get('savename', 'best_model.pth')

    m = model(p) # Khởi tạo Controller mô hình

    # Khối logic huấn luyện
    if force_train or p.get('rebuild', False):
        train_data, val_data = train, val
        # Load data nếu chưa truyền vào
        if train_data is None or val_data is None:
            train_data, val_data = load_data_catsVsdogs(p)

        # Số epoch huấn luyện bổ sung (nếu train tiếp)
        add_epochs = int(p.get('epochs', 0))

        # Thử load checkpoint cũ để huấn luyện tiếp (Resume Training)
        try:
            m.load_model()
            print(f"🔁 Resume training from existing best checkpoint for {p['architecture']}")

            # Tính lại đích đến của vòng lặp: Epoch đang đứng + Số epoch thêm vào
            target = m.start_epoch + add_epochs
            if target <= m.start_epoch:
                target = m.start_epoch
            m.params['epochs'] = target
            print(f"🔢 Will train additional {add_epochs} epochs -> total epochs target: {m.params['epochs']}")
        except Exception:
            # Nếu không load được, train mới hoàn toàn
            print(f"⚠️ Không có checkpoint hợp lệ cho {p['architecture']} → huấn luyện mới")
            m.params['epochs'] = add_epochs

        m.converge(train_data, val_data) # Bắt đầu vòng lặp huấn luyện

        # Sau khi train xong, load lại trọng số tốt nhất đã đạt được
        try:
            m.load_model()
            print(f"✅ Training for {p['architecture']} done (best model loaded)")
        except Exception as e:
            print(f"✅ Training for {p['architecture']} done (but failed to load best model: {e})")
    else:
        # Nếu không yêu cầu train (force_train=False), chỉ cần load model
        try:
            m.load_model()
        except Exception as e:
            # Nếu mất file model, bắt buộc phải train lại
            print(f"⚠️ Failed to load model for {p['architecture']}: {e} — will train instead")
            train_data, val_data = load_data_catsVsdogs(p)
            m.converge(train_data, val_data)
            try:
                m.load_model()
                print(f"✅ Training for {p['architecture']} done (best model loaded)")
            except Exception as e:
                print(f"✅ Training for {p['architecture']} done (but failed to load best model: {e})")

    val_loss = None
    val_acc = None
    try:
        if val is None:
            _, val = load_data_catsVsdogs(p)
        # Tiến hành đánh giá trên tập validation
        val_loss, val_acc = m.evaluate(val)
        print(f"Validation loss for {p['architecture']}: {val_loss:.4f} | Acc: {val_acc:.4f}")
    except Exception as e:
        print(f"⚠️ Could not evaluate {p['architecture']}: {e}")

    # Chạy dự đoán cho 1 folder ảnh riêng lẻ nếu được yêu cầu
    results = None
    if test_folder_path is not None:
        try:
            results = m.test_folder(test_folder_path)
            heatmap_dir = os.path.join(p.get('savedir', '.'), 'heatmaps')
            if os.path.exists(heatmap_dir) and len(os.listdir(heatmap_dir)) > 0:
                print(f"✅ Đã lưu ảnh Heatmap tại: {heatmap_dir}")
                ans = input("Mở thư mục chứa ảnh Heatmap để xem? (y/n) [n]: ").strip().lower()
                if ans == 'y':
                    try:
                        os.startfile(heatmap_dir)
                    except Exception:
                        pass
        except Exception as e:
            print(f"⚠️ Failed to run test_folder on {test_folder_path}: {e}")

    return val_loss, m, results


# Lấy đường dẫn tuyệt đối của thư mục test_images dựa trên vị trí file code
TEST_DIR = str(pathlib.Path(__file__).parent / 'test_images')

# Hàm giao diện dòng lệnh (CLI)
def main():
    while True:
        # In menu ra màn hình
        print("\nChọn chức năng:")
        print("1) Huấn luyện CNN")
        print("2) Huấn luyện CNN + SE")
        print("3) Huấn luyện cả 2 (CNN và CNN+SE)")  
        print("4) Dùng model CNN để phân loại ảnh trong thư mục test_images")
        print("5) Dùng model CNN+SE để phân loại ảnh trong thư mục test_images")
        print("6) Tìm ảnh trên Google và tải về test_images")
        print("7) Phân loại ảnh trong test_images (chọn model)")
        print("8) Xóa tất cả ảnh trong thư mục test_images")
        print("9) Huấn luyện ResNet18")
        print("10) Huấn luyện ResNet18 + SE")
        print("11) Dùng ResNet18 phân loại ảnh trong test_images")
        print("12) Dùng ResNet18 + SE phân loại ảnh trong test_images")
        print("13) Huấn luyện cả 2 (ResNet và ResNet+SE)")
        print("0) Thoát")

        choice = input("Enter choice: ").strip() # Nhận lệnh từ người dùng

        if choice == '0':
            print("Exiting.")
            break

        # Tùy chọn 1: Train CNN thường
        elif choice == '1':
            e = input(f"Số epoch để huấn luyện (số epoch bổ sung, mặc định {parameters.get('epochs',5)}): ").strip()
            try:
                e = int(e) if e else parameters.get('epochs', 5)
            except ValueError:
                e = parameters.get('epochs', 5)

            train, val = load_data_catsVsdogs(parameters) # Tải dữ liệu
            p = copy.deepcopy(parameters)
            p['epochs'] = e
            run_for_arch(p, 'cnn', train, val, force_train=True, test_folder_path=None) # Chạy train

        # Tùy chọn 2: Train SE-CNN
        elif choice == '2':
            e = input(f"Số epoch để huấn luyện (số epoch bổ sung, mặc định {parameters.get('epochs',5)}): ").strip()
            try:
                e = int(e) if e else parameters.get('epochs', 5)
            except ValueError:
                e = parameters.get('epochs', 5)

            train, val = load_data_catsVsdogs(parameters)
            p = copy.deepcopy(parameters)
            p['epochs'] = e
            run_for_arch(p, 'se', train, val, force_train=True, test_folder_path=None)

        # Tùy chọn 3: Train cả 2 liên tiếp
        elif choice == '3':
            print("📥 Loading dataset once for both experiments...")
            e = input(f"Số epoch để huấn luyện cả hai mô hình (số epoch bổ sung, mặc định {parameters.get('epochs',5)}): ").strip()
            try:
                e = int(e) if e else parameters.get('epochs', 5)
            except ValueError:
                e = parameters.get('epochs', 5)

            train, val = load_data_catsVsdogs(parameters)
            p1 = copy.deepcopy(parameters)
            p1['epochs'] = e
            p2 = copy.deepcopy(parameters)
            p2['epochs'] = e

            run_for_arch(p1, 'cnn', train, val, force_train=True, test_folder_path=None)
            run_for_arch(p2, 'se', train, val, force_train=True, test_folder_path=None)

        # Tùy chọn 13: Train ResNet & ResNet+SE liên tiếp
        elif choice == '13':
            print("📥 Loading dataset once for both ResNet experiments...")
            e = input(f"Số epoch để huấn luyện cả hai mô hình ResNet (số epoch bổ sung, mặc định {parameters.get('epochs',5)}): ").strip()
            try:
                e = int(e) if e else parameters.get('epochs', 5)
            except ValueError:
                e = parameters.get('epochs', 5)

            train, val = load_data_catsVsdogs(parameters)

            # Cấu hình ResNet
            p1 = copy.deepcopy(parameters)
            p1['savedir'] = f"{p1.get('savedir')}_resnet"
            p1['epochs'] = e
            trainer1 = ResNetTrainer(p1, use_se=False)

            # Cấu hình ResNet+SE
            p2 = copy.deepcopy(parameters)
            p2['savedir'] = f"{p2.get('savedir')}_resnet_se"
            p2['epochs'] = e
            trainer2 = ResNetTrainer(p2, use_se=True)

            trainer1.converge(train, val) # Train mạng 1
            trainer2.converge(train, val) # Train mạng 2

        # Tùy chọn 9: Train ResNet
        elif choice == '9':
            e = input(f"Số epoch để huấn luyện ResNet (số epoch bổ sung, mặc định {parameters.get('epochs',5)}): ").strip()
            try:
                e = int(e) if e else parameters.get('epochs', 5)
            except ValueError:
                e = parameters.get('epochs', 5)

            p = copy.deepcopy(parameters)
            p['savedir'] = f"{p.get('savedir')}_resnet"
            p['epochs'] = e
            trainer = ResNetTrainer(p, use_se=False)
            train_dl, val_dl = load_data_catsVsdogs(p)
            trainer.converge(train_dl, val_dl)

        # Tùy chọn 10: Train ResNet+SE
        elif choice == '10':
            e = input(f"Số epoch để huấn luyện ResNet+SE (số epoch bổ sung, mặc định {parameters.get('epochs',5)}): ").strip()
            try:
                e = int(e) if e else parameters.get('epochs', 5)
            except ValueError:
                e = parameters.get('epochs', 5)

            p = copy.deepcopy(parameters)
            p['savedir'] = f"{p.get('savedir')}_resnet_se"
            p['epochs'] = e
            trainer = ResNetTrainer(p, use_se=True)
            train_dl, val_dl = load_data_catsVsdogs(p)
            trainer.converge(train_dl, val_dl)

        # Tùy chọn 11: Test ResNet trên ảnh lẻ
        elif choice == '11':
            p = copy.deepcopy(parameters)
            p['savedir'] = f"{p.get('savedir')}_resnet"
            trainer = ResNetTrainer(p, use_se=False)
            try:
                trainer.load_model()
                trainer.test_folder(TEST_DIR)
                
                heatmap_dir = os.path.join(p['savedir'], 'heatmaps')
                if os.path.exists(heatmap_dir) and len(os.listdir(heatmap_dir)) > 0:
                    print(f"✅ Đã lưu ảnh Heatmap tại: {heatmap_dir}")
                    ans = input("Mở thư mục chứa ảnh Heatmap để xem? (y/n) [n]: ").strip().lower()
                    if ans == 'y':
                        try:
                            os.startfile(heatmap_dir)
                        except Exception:
                            pass
            except Exception as e:
                print(f"⚠️ Không load được ResNet model: {e}")

        # Tùy chọn 12: Test ResNet+SE trên ảnh lẻ
        elif choice == '12':
            p = copy.deepcopy(parameters)
            p['savedir'] = f"{p.get('savedir')}_resnet_se"
            trainer = ResNetTrainer(p, use_se=True)
            try:
                trainer.load_model()
                trainer.test_folder(TEST_DIR)

                heatmap_dir = os.path.join(p['savedir'], 'heatmaps')
                if os.path.exists(heatmap_dir) and len(os.listdir(heatmap_dir)) > 0:
                    print(f"✅ Đã lưu ảnh Heatmap tại: {heatmap_dir}")
                    ans = input("Mở thư mục chứa ảnh Heatmap để xem? (y/n) [n]: ").strip().lower()
                    if ans == 'y':
                        try:
                            os.startfile(heatmap_dir)
                        except Exception:
                            pass
            except Exception as e:
                print(f"⚠️ Không load được ResNet+SE model: {e}")

        # Tùy chọn 4: Phân loại bằng CNN
        elif choice == '4':
            run_for_arch(parameters, 'cnn', None, None, force_train=False, test_folder_path=TEST_DIR)

        # Tùy chọn 5: Phân loại bằng SE-CNN
        elif choice == '5':
            run_for_arch(parameters, 'se', None, None, force_train=False, test_folder_path=TEST_DIR)

        # Tùy chọn 6: Tải ảnh từ internet về thư mục test
        elif choice == '6':
            q = input("Nhập từ khoá tìm kiếm (ví dụ: cat, dog) [cat]: ").strip() or 'cat'

            dest = pathlib.Path(TEST_DIR)
            dest.mkdir(parents=True, exist_ok=True) # Tạo thư mục nếu chưa có

            print(f"Mở Google Images với từ khoá: {q}. Vui lòng chọn và tải ảnh thủ công vào thư mục '{dest}'.")
            # Mở trình duyệt web
            webbrowser.open(f"https://www.google.com/search?tbm=isch&q={urllib.request.quote(q)}")

            input(f"Khi bạn đã tải xong ảnh vào thư mục '{TEST_DIR}', nhấn Enter để quay lại menu...")

        # Tùy chọn 8: Dọn dẹp thư mục test_images
        elif choice == '8':
            dest = pathlib.Path(TEST_DIR)
            if not dest.exists():
                print(f"Thư mục '{TEST_DIR}' không tồn tại.")
                continue

            # Liệt kê tất cả file ảnh
            imgs = [p for p in dest.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')]
            if not imgs:
                print(f"Không có ảnh nào trong '{TEST_DIR}'.")
                continue

            print(f"Tìm thấy {len(imgs)} ảnh trong 'test_images'.")
            confirm = input("Bạn có chắc muốn xóa tất cả các ảnh này? (y/N): ").strip().lower()
            if confirm == 'y':
                removed = 0
                for p in imgs:
                    try:
                        p.unlink() # Xóa file
                        removed += 1
                    except Exception as e:
                        print(f"Không xóa được {p}: {e}")

                print(f"Đã xóa {removed} file ảnh khỏi 'test_images'.")
            else:
                print("Hủy thao tác xóa.")

        # Tùy chọn 7: Tùy biến phân loại bằng CNN/SE-CNN
        elif choice == '7':
            print("Choose model to classify test_images:\n1) CNN\n2) CNN+SE\n3) Both")
            c = input("Enter 1/2/3: ").strip() or '1'
            if c == '1':
                _, _, results = run_for_arch(parameters, 'cnn', None, None, force_train=False, test_folder_path='test_images')
                if results:
                    print('\nTổng kết phân loại (CNN):')
                    for fn, label, cat_p, dog_p in results:
                        print(f"{fn:20} -> {('🐱 Cat' if label=='cat' else '🐶 Dog')} | Cat={cat_p:.2f} Dog={dog_p:.2f}")
            elif c == '2':
                _, _, results = run_for_arch(parameters, 'se', None, None, force_train=False, test_folder_path='test_images')
                if results:
                    print('\nTổng kết phân loại (SE-CNN):')
                    for fn, label, cat_p, dog_p in results:
                        print(f"{fn:20} -> {('🐱 Cat' if label=='cat' else '🐶 Dog')} | Cat={cat_p:.2f} Dog={dog_p:.2f}")
            elif c == '3':
                print("Running CNN predictions:")
                _, _, results1 = run_for_arch(parameters, 'cnn', None, None, force_train=False, test_folder_path='test_images')
                print("Running SE-CNN predictions:")
                _, _, results2 = run_for_arch(parameters, 'se', None, None, force_train=False, test_folder_path='test_images')

                if results1:
                    print('\nTổng kết phân loại (CNN):')
                    for fn, label, cat_p, dog_p in results1:
                        print(f"{fn:20} -> {('🐱 Cat' if label=='cat' else '🐶 Dog')} | Cat={cat_p:.2f} Dog={dog_p:.2f}")

                if results2:
                    print('\nTổng kết phân loại (SE-CNN):')
                    for fn, label, cat_p, dog_p in results2:
                        print(f"{fn:20} -> {('🐱 Cat' if label=='cat' else '🐶 Dog')} | Cat={cat_p:.2f} Dog={dog_p:.2f}")
            else:
                print("Invalid choice")

        else:
            print("Invalid choice, try again.")


if __name__ == '__main__':
    main()

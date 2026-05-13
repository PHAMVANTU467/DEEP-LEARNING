# app.py
"""
Web UI cho hệ thống phân loại chó/mèo sử dụng Streamlit.
- Tách biệt frontend (UI) và backend (logic)
- Import lại toàn bộ code xử lý từ các file hiện có
- KHÔNG thay đổi thuật toán/model/logic
"""
import streamlit as st # Thư viện tạo giao diện web bằng Python nhanh chóng
import sys # Thư viện hệ thống để can thiệp vào đường dẫn Python
import os # Thư viện xử lý file và thư mục
import pathlib # Thư viện thao tác với đường dẫn an toàn hơn
import importlib # Thư viện nạp lại module bằng chuỗi văn bản
import time # Thư viện xử lý thời gian
import threading # Thư viện chạy đa luồng (multi-threading) để giao diện không bị đơ khi train
import traceback # Thư viện in ra lỗi chi tiết
import json # Thư viện xử lý file json
from PIL import Image # Thư viện xử lý ảnh
from typing import List, Tuple # Thư viện gợi ý kiểu dữ liệu
import torch.nn.functional as F # Thư viện hàm Pytorch
import torch # Import framework Pytorch
import warnings # Thư viện xử lý cảnh báo

# Tối ưu hoá luồng tính toán Convolution trên GPU (Tăng 10-20% tốc độ mà không giảm Acc)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Ẩn các cảnh báo làm phiền trên terminal
warnings.filterwarnings('ignore', message='.*Truncated File Read.*')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.TiffImagePlugin')

# Cấu hình UI/UX
st.set_page_config(page_title="AI Vision: Phân Loại Chó/Mèo", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# CSS tùy chỉnh đã bị tắt để tránh lỗi DOM trên Streamlit Cloud
# st.markdown("""
# <style>
#     /* CSS Tùy chỉnh cho trải nghiệm người dùng cao cấp */
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
#     h1, h2, h3 {
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         color: #1E3A8A;
#         font-weight: 700;
#     }
#     .stButton>button {
#         border-radius: 8px;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
#     .stButton>button[kind="primary"] {
#         background: linear-gradient(135deg, #2563EB, #1D4ED8);
#         border: none;
#         color: white;
#         box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
#     }
#     .stButton>button[kind="primary"]:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
#     }
#     .stProgress .st-bo {
#         background-color: #2563EB;
#     }
#     div[data-testid="stMetricValue"] {
#         font-size: 1.8rem;
#         color: #047857;
#     }
# </style>
# """, unsafe_allow_html=True)

# Thêm thư mục gốc (project root) vào sys.path để import các module từ src
import sys
import copy
import traceback

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    add_script_run_ctx = None

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import src.core.tools as tools_mod # Tải module tools
import src.models.cnn_trainer as cnn_mod # Tải module model (CNN Controller)
import src.models.resnet_trainer as resnet_mod # Tải module resnet
import src.core.utils_run as run_arch_mod # Tải quy trình train

# Lấy các hàm/lớp quan trọng từ các module vừa tải
load_data_catsVsdogs = tools_mod.load_data_catsVsdogs
size_conv_output = tools_mod.size_conv_output
ResNetTrainer = resnet_mod.ResNetTrainer
load_cnn_model = cnn_mod.load_cnn_model
load_resnet_model = resnet_mod.load_resnet_model
load_resnet_se_model = resnet_mod.load_resnet_se_model
run_for_arch = run_arch_mod.run_for_arch
from src.app.inference import ModelManager, Predictor, HeatmapGenerator # Import module pipeline dự đoán

# Cấu hình danh sách các Model có trong hệ thống (Tên hiển thị, key code)
MODEL_OPTIONS = [
    ("CNN", "cnn"),
    ("CNN+SE", "se"),
    ("ResNet18", "resnet"),
    ("ResNet18+SE", "resnet_se")
]
# Tạo từ điển map 2 chiều giữa tên và key
MODEL_LABEL_TO_KEY = {label: key for label, key in MODEL_OPTIONS}
MODEL_KEY_TO_LABEL = {key: label for label, key in MODEL_OPTIONS}
MODEL_LABELS = [label for label, _ in MODEL_OPTIONS]

TEST_IMAGES_DIR = "data/test_images" # Thư mục chứa ảnh dùng thử
@st.cache_resource
def get_train_state_lock():
    return threading.Lock() # Khóa (Lock) để tránh xung đột dữ liệu khi chạy đa luồng

@st.cache_resource
def get_train_state():
    # Biến trạng thái toàn cục lưu trữ thông tin về quá trình huấn luyện đang chạy ngầm
    return {
        "running": False, # Cờ báo hiệu có tiến trình train đang chạy không
        "stop_event": None, # Sự kiện dùng để huỷ train giữa chừng
        "skip_event": None, # Cờ bỏ qua model hiện tại
        "worker": None, # Thread chạy ngầm
        "selected_model_keys": [], # Danh sách model được chọn train
        "current_model_key": None, # Model đang train hiện tại
        "current_model_index": 0, # Số thứ tự
        "total_models": 0, # Tổng số model cần train
        "current_epoch": 0, # Vòng lặp epoch hiện tại
        "target_epoch": 0, # Tổng epoch cần đạt
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
        "batch_done": 0,
        "batch_total": 0,
        "epoch_started_at": 0,
        "started_at": None, # Thời điểm bắt đầu
        "elapsed": 0.0, # Thời gian đã trôi qua
        "logs": [], # Dòng chữ thông báo log
        "error": None, # Lưu lỗi nếu có
        "stopped": False, # Bị dừng bởi người dùng?
        "done": False, # Hoàn thành chưa?
        "needs_cache_clear": False, # Cờ yêu cầu xóa cache để model UI nhận bản train mới nhất
    }

TRAIN_STATE_LOCK = get_train_state_lock()
TRAIN_STATE = get_train_state()


# ============= CÁC HÀM CACHE ĐỂ GIAO DIỆN CHẠY NHANH =============
@st.cache_resource # Streamlit sẽ cache lại object này vào RAM, không gọi lại ở lần reload UI sau
def get_model_manager(params: dict):
    return ModelManager(params) # Trả về trình quản lý model


@st.cache_resource
def get_predictor(params: dict):
    manager = get_model_manager(params)
    return Predictor(manager) # Trả về trình dự đoán


@st.cache_resource
def get_heatmap_generator():
    return HeatmapGenerator() # Trả về trình vẽ heatmap


@st.cache_resource
def load_model_cached(model_key: str, params: dict):
    # Hàm load 1 mô hình vào RAM và lưu lại
    manager = get_model_manager(params)
    wrapped_model = manager.get_model(model_key)
    wrapped_model.torch_model().eval() # Chuyển sang Eval mode
    return wrapped_model


# Hàm lấy tên file trọng số của 1 model
def _checkpoint_candidates(model_key: str, params: dict) -> List[str]:
    checkpoint_root = params.get("checkpoint_root", "checkpoints")
    dataset_name = params.get("dataset_name", "dataset_80_20")
    return [os.path.join(checkpoint_root, dataset_name, model_key, "best_model.pth")]


# Hàm băm (hash) trạng thái của file trọng số (để check xem file có bị thay đổi sau khi train chưa)
def _model_signature(model_keys: List[str], params: dict) -> Tuple[Tuple[str, str, int], ...]:
    signature = []
    for model_key in sorted(set(model_keys)):
        found_path = ""
        mtime_ns = -1
        # Duyệt tìm file
        for candidate in _checkpoint_candidates(model_key, params):
            if os.path.exists(candidate):
                found_path = candidate
                mtime_ns = os.stat(candidate).st_mtime_ns # Lấy thời gian sửa đổi file
                break
        signature.append((model_key, found_path, mtime_ns))
    return tuple(signature)


# Hàm kiểm tra và xóa RAM (cache) nếu phát hiện file trọng số thay đổi (có bản train mới)
def ensure_model_cache_fresh(model_keys: List[str], params: dict, state_key: str):
    current = _model_signature(model_keys, params)
    previous = st.session_state.get(state_key) # Lấy lịch sử check cũ
    if previous is None:
        st.session_state[state_key] = current
        return
    if previous != current: # Khác nhau nghĩa là model vừa cập nhật!
        st.cache_resource.clear() # Xóa hết RAM đang lưu model cũ
        st.session_state[state_key] = current
        st.info("Phát hiện file model thay đổi, cache đã được reload tự động.")


# Hàm bấm tay xóa cache
def clear_model_caches(success_message: str = "Models reloaded"):
    st.cache_resource.clear()
    st.session_state["predict_model_signature"] = None
    st.session_state["viz_model_signature"] = None
    st.session_state["eval_model_signature"] = None
    st.success(success_message)


# Đếm tổng số lượng tham số trong mạng nơ-ron
def count_params(model):
    return sum(p.numel() for p in model.parameters())


# Hàm ghi chú (log) quá trình train an toàn khi dùng luồng (Thread-safe)
def _append_train_log(message: str):
    timestamp = time.strftime("%H:%M:%S") # Lấy giờ hệ thống
    with TRAIN_STATE_LOCK: # Khóa lại để luồng khác khỏi ghi đè
        TRAIN_STATE["logs"].append(f"[{timestamp}] {message}")
        TRAIN_STATE["logs"] = TRAIN_STATE["logs"][-300:] # Chỉ giữ 300 dòng mới nhất


# Lấy bản sao trạng thái train để in lên màn hình UI
def _get_train_state_snapshot() -> dict:
    with TRAIN_STATE_LOCK:
        stop_event = TRAIN_STATE.get("stop_event")
        return {
            "running": TRAIN_STATE["running"],
            "selected_model_keys": list(TRAIN_STATE["selected_model_keys"]),
            "current_model_key": TRAIN_STATE["current_model_key"],
            "current_model_index": TRAIN_STATE["current_model_index"],
            "total_models": TRAIN_STATE["total_models"],
            "current_epoch": TRAIN_STATE["current_epoch"],
            "target_epoch": TRAIN_STATE["target_epoch"],
            "train_loss": TRAIN_STATE.get("train_loss", 0.0),
            "val_loss": TRAIN_STATE.get("val_loss", 0.0),
            "val_acc": TRAIN_STATE.get("val_acc", 0.0),
            "batch_done": TRAIN_STATE.get("batch_done", 0),
            "batch_total": TRAIN_STATE.get("batch_total", 0),
            "epoch_started_at": TRAIN_STATE.get("epoch_started_at", 0),
            "started_at": TRAIN_STATE["started_at"],
            "elapsed": TRAIN_STATE["elapsed"],
            "logs": list(TRAIN_STATE["logs"]),
            "error": TRAIN_STATE["error"],
            "stopped": TRAIN_STATE["stopped"],
            "done": TRAIN_STATE["done"],
            "needs_cache_clear": TRAIN_STATE["needs_cache_clear"],
            "stop_requested": bool(stop_event.is_set()) if stop_event is not None else False,
        }


# Gửi tín hiệu ngắt vòng lặp huấn luyện nếu người dùng bấm nút Dừng
def _request_stop_training():
    with TRAIN_STATE_LOCK:
        stop_event = TRAIN_STATE.get("stop_event")
    if stop_event is not None:
        stop_event.set() # Bật cờ dừng
        _append_train_log("Đã nhận yêu cầu dừng huấn luyện.")


def _request_skip_model():
    with TRAIN_STATE_LOCK:
        skip_event = TRAIN_STATE.get("skip_event")
    if skip_event is not None:
        skip_event.set()
        _append_train_log("Đã nhận yêu cầu BỎ QUA model hiện tại.")


# Đánh dấu đã xóa cache
def _ack_cache_clear_done():
    with TRAIN_STATE_LOCK:
        TRAIN_STATE["needs_cache_clear"] = False


# ============= THREAD HUẤN LUYỆN (Chạy ngầm để không đơ UI) =============
def _training_worker(selected_model_keys: List[str], epochs: int, ignore_checkpoint: bool):
    stop_event = None
    dataset_name = ""
    with TRAIN_STATE_LOCK:
        stop_event = TRAIN_STATE["stop_event"]
        dataset_name = TRAIN_STATE.get("dataset_name", "")

    total_start = time.time()
    try:
        parameters = get_default_parameters() # Lấy cấu hình mặc định
        parameters['epochs'] = int(epochs) # Cập nhật epoch
        parameters['dataset_name'] = dataset_name # Truyền tên bộ dữ liệu vào param

        # Nếu người dùng yêu cầu xóa file trọng số cũ để train trắng
        if ignore_checkpoint:
            for model_key in selected_model_keys:
                model_dir = os.path.join('checkpoints', dataset_name, model_key)
                checkpoint_path = os.path.join(model_dir, 'best_model.pth')
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path) # Xóa file cũ
                    _append_train_log(f"Đã xóa checkpoint: {checkpoint_path}")
                history_path = os.path.join(model_dir, 'history.json')
                if os.path.exists(history_path):
                    os.remove(history_path)
                    _append_train_log(f"Đã xóa history: {history_path}")

        # Lặp qua các mô hình người dùng yêu cầu train
        for index, model_key in enumerate(selected_model_keys, start=1):
            if stop_event is not None and stop_event.is_set():
                _append_train_log("Dừng theo yêu cầu người dùng trước khi bắt đầu model tiếp theo.")
                break

            model_name = MODEL_KEY_TO_LABEL[model_key]
            with TRAIN_STATE_LOCK:
                TRAIN_STATE["current_model_key"] = model_key
                TRAIN_STATE["current_model_index"] = index
                TRAIN_STATE["current_epoch"] = 0
                TRAIN_STATE["target_epoch"] = int(epochs)

            _append_train_log(f"Bắt đầu train {model_name} ({index}/{len(selected_model_keys)})")

            # Reset cờ skip cho model mới
            with TRAIN_STATE_LOCK:
                skip_event = TRAIN_STATE.get("skip_event")
                if skip_event is not None:
                    skip_event.clear()

            # Callback hàm gọi ngược để kiểm tra cờ Dừng ở tầng sâu bên trong model.py
            def _stop_requested() -> bool:
                with TRAIN_STATE_LOCK:
                    se = TRAIN_STATE.get("skip_event")
                    skip_triggered = se.is_set() if se else False
                return bool((stop_event is not None and stop_event.is_set()) or skip_triggered)

            # Callback gọi ngược để cập nhật thanh tiến trình % UI
            def _progress_callback(done_epoch: int, target_epoch: int, _train_loss: float, _val_loss: float, _val_acc: float, batch_done: int = 0, batch_total: int = 0):
                with TRAIN_STATE_LOCK:
                    TRAIN_STATE["current_epoch"] = int(done_epoch) if done_epoch is not None else TRAIN_STATE["current_epoch"]
                    TRAIN_STATE["target_epoch"] = int(target_epoch) if target_epoch is not None else TRAIN_STATE["target_epoch"]
                    if _train_loss is not None:
                        TRAIN_STATE["train_loss"] = float(_train_loss)
                    if _val_loss is not None:
                        TRAIN_STATE["val_loss"] = float(_val_loss)
                    if _val_acc is not None:
                        TRAIN_STATE["val_acc"] = float(_val_acc)
                    if batch_done is not None:
                        if int(batch_done) == 0:
                            TRAIN_STATE["epoch_started_at"] = time.time()
                        TRAIN_STATE["batch_done"] = int(batch_done)
                    if batch_total is not None:
                        TRAIN_STATE["batch_total"] = int(batch_total)

            try:
                # Load dữ liệu dataset
                train, val = load_data_catsVsdogs(parameters)

                # Gọi hàm converge tương ứng cho CNN hoặc ResNet
                if model_key in {'cnn', 'se'}:
                    from src.models.cnn_trainer import load_cnn_model
                    use_se = (model_key == "se")
                    trainer = load_cnn_model(parameters, use_se=use_se, load_optimizer=True)
                    trainer.converge(
                        train,
                        val,
                        stop_requested=_stop_requested,
                        progress_callback=_progress_callback,
                    )
                elif "resnet" in model_key:
                    from src.models.resnet_trainer import load_resnet_model, load_resnet_se_model
                    if model_key == "resnet":
                        trainer = load_resnet_model(parameters, load_optimizer=True)
                    else:
                        trainer = load_resnet_se_model(parameters, load_optimizer=True)
                    trainer.converge(
                        train,
                        val,
                        stop_requested=_stop_requested,
                        progress_callback=_progress_callback,
                    )

                with TRAIN_STATE_LOCK:
                    skip_evt = TRAIN_STATE.get("skip_event")
                    is_skipped = skip_evt.is_set() if skip_evt else False
                    
                if is_skipped:
                    _append_train_log(f"{model_name}: đã BỎ QUA.")
                elif stop_event is not None and stop_event.is_set():
                    _append_train_log(f"{model_name}: đã dừng theo yêu cầu.")
                    break
                else:
                    _append_train_log(f"{model_name}: hoàn tất.")
            except Exception as model_exc:
                _append_train_log(f"{model_name}: lỗi - {model_exc}")
                _append_train_log(traceback.format_exc())

        # Sau khi lặp xong các model
        stopped = bool(stop_event is not None and stop_event.is_set())
        with TRAIN_STATE_LOCK:
            TRAIN_STATE["stopped"] = stopped
            TRAIN_STATE["done"] = True # Cờ hoàn thành
            TRAIN_STATE["elapsed"] = time.time() - total_start
            TRAIN_STATE["needs_cache_clear"] = True # Cờ báo hiệu UI tải lại model mới
    except Exception as exc:
        # Nếu luồng bị crash (như hết RAM, lỗi code)
        with TRAIN_STATE_LOCK:
            TRAIN_STATE["error"] = str(exc)
            TRAIN_STATE["done"] = True
            TRAIN_STATE["elapsed"] = time.time() - total_start
            TRAIN_STATE["stopped"] = bool(stop_event is not None and stop_event.is_set())
            TRAIN_STATE["needs_cache_clear"] = True
        _append_train_log(traceback.format_exc())
    finally:
        with TRAIN_STATE_LOCK:
            TRAIN_STATE["running"] = False # Đánh dấu kết thúc


# Hàm khởi động tiến trình đa luồng (gọi khi người dùng bấm nút)
def _start_training_job(selected_model_keys: List[str], epochs: int, ignore_checkpoint: bool, dataset_name: str):
    stop_event = threading.Event()
    skip_event = threading.Event()
    with TRAIN_STATE_LOCK:
        # Khởi tạo trạng thái mới tinh cho một quá trình train
        TRAIN_STATE["running"] = True
        TRAIN_STATE["stop_event"] = stop_event
        TRAIN_STATE["skip_event"] = skip_event
        TRAIN_STATE["dataset_name"] = dataset_name
        TRAIN_STATE["worker"] = None
        TRAIN_STATE["selected_model_keys"] = list(selected_model_keys)
        TRAIN_STATE["current_model_key"] = None
        TRAIN_STATE["current_model_index"] = 0
        TRAIN_STATE["total_models"] = len(selected_model_keys)
        TRAIN_STATE["current_epoch"] = 0
        TRAIN_STATE["target_epoch"] = int(epochs)
        TRAIN_STATE["train_loss"] = 0.0
        TRAIN_STATE["val_loss"] = 0.0
        TRAIN_STATE["val_acc"] = 0.0
        TRAIN_STATE["started_at"] = time.time()
        TRAIN_STATE["elapsed"] = 0.0
        TRAIN_STATE["logs"] = []
        TRAIN_STATE["error"] = None
        TRAIN_STATE["stopped"] = False
        TRAIN_STATE["done"] = False
        TRAIN_STATE["needs_cache_clear"] = False

    # Khởi tạo luồng Thread
    worker = threading.Thread(
        target=_training_worker,
        args=(list(selected_model_keys), int(epochs), bool(ignore_checkpoint)),
        daemon=True, # Chạy ngầm, tự chết nếu ứng dụng chính tắt
    )
    if add_script_run_ctx:
        add_script_run_ctx(worker)
        
    with TRAIN_STATE_LOCK:
        TRAIN_STATE["worker"] = worker
        
    worker.start() # Bắt đầu chạy


# Load model để đánh giá (evaluate tab)
@st.cache_resource
def load_eval_model_cached(model_key: str, params: dict, _val_loader=None):
    # Keep evaluate in no-retrain mode; if fallback happens in run_for_arch it runs 0 epoch.
    eval_params = params.copy()
    eval_params["epochs"] = 0 # Ép epoch bằng 0 để không bị chạy nhầm logic train

    # Khởi tạo từng đối tượng model khác nhau
    if model_key in {"cnn", "se"}:
        trainer_obj = load_cnn_model(eval_params, use_se=(model_key == "se"), load_optimizer=False)
    elif model_key in {"resnet", "resnet_se"}:
        if model_key == "resnet":
            trainer_obj = load_resnet_model(eval_params, load_optimizer=False)
        else:
            trainer_obj = load_resnet_se_model(eval_params, load_optimizer=False)
    else:
        raise ValueError(f"Model key không hợp lệ: {model_key}")

    trainer_obj.model.eval() # Bật chế độ đánh giá cho mô hình
    return trainer_obj


# Thu thập toàn bộ nhãn dự đoán để tính toán metric phức tạp (Precision/Recall/F1)
def collect_predictions(model_key: str, trainer_obj, val_loader, input_size: int):
    y_true = [] # Mảng lưu nhãn thật
    y_pred = [] # Mảng lưu nhãn mà model dự đoán
    device = trainer_obj.model.device if hasattr(trainer_obj.model, "device") else getattr(trainer_obj, "device", "cpu")

    trainer_obj.model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            # Reshape chuẩn 3 kênh RGB
            x = x.view(-1, 3, input_size, input_size)

            logits = trainer_obj.model(x)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    return y_true, y_pred


# ============= GIAO DIỆN (UI) VỚI STREAMLIT =============
is_preprocessing = st.session_state.get("is_preprocessing", False)
is_evaluating = st.session_state.get("is_evaluating", False)

# Lấy trạng thái từ thread huấn luyện
train_state = _get_train_state_snapshot()
is_training = train_state.get("running", False)

# Biến cờ tổng hợp: Nếu có bất kỳ tác vụ nặng nào đang chạy
is_processing_anything = is_preprocessing or is_evaluating or is_training

# Sidebar (Cột điều hướng bên trái màn hình)
with st.sidebar:
    st.title("🐾 AI Vision Menu")
    st.markdown("Hệ thống quản lý vòng đời huấn luyện")
    st.divider()

    if is_processing_anything:
        st.error("⚠️ HỆ THỐNG ĐANG BẬN\n\nCác tab điều hướng đã bị khóa tạm thời để đảm bảo an toàn cho tiến trình đang chạy. Bạn có thể sử dụng nút 'Dừng khẩn cấp' nếu thực sự muốn chuyển tab.")

    page = st.radio("CÁC CHỨC NĂNG CHÍNH", [
        "Tiền xử lý Dữ liệu",
        "Huấn luyện",
        "Biểu đồ Lịch sử",
        "Dự đoán & Phân tích",
        "Đánh giá",
        "Quản lý dữ liệu"
    ], disabled=is_processing_anything)
    st.divider()
    st.caption("🚀 Antigravity - Advanced Agentic UI")

# Hàm khởi tạo dictionary cấu hình chung mặc định (thay vì code cứng)
def get_default_parameters():
    parameters = {}
    parameters['path_cats'] = 'data/PetImages/Cat'
    parameters['path_dogs'] = 'data/PetImages/Dog'
    parameters['size'] = 64 # Giảm xuống 64 để CPU xử lý mượt mà (Tăng tốc gấp nhiều lần)
    parameters['input_channel'] = 3
    parameters['conv_output_channels'] = [32,64,128]
    parameters['conv_kernel_size'] = [5,5,5]
    parameters['stride_size'] = [1,1,1]
    parameters['padding_size'] = [0,0,0]
    parameters['pool_kernel_size'] = [2,2,2]
    parameters['pool_stride_size'] = [2,2,2]
    parameters['pool_padding_size'] = [0,0,0]
    parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    parameters['output_dimen_cnn'] = size_conv_output(parameters)
    parameters['# inputs'] = parameters['output_dimen_cnn']
    parameters['# outputs'] = 2
    parameters['validation'] = 0.1
    parameters['training'] = 0.9
    parameters['batch_size_training'] = 64 # Đưa về 64 cho CPU
    parameters['batch_size_validation'] = 64
    parameters['learning rate'] = 1e-3
    parameters['epochs'] = 5
    parameters['savedir'] = 'checkpoints'
    parameters['savename'] = 'best_model.pth'
    parameters['dataset'] = 'data/dataset.npy'
    return parameters

import shutil

# 0. GIAO DIỆN TAB TIỀN XỬ LÝ
def preprocess_tab():
    st.header("🛠️ Tiền xử lý & Chia tập dữ liệu (Data Preprocessing)")
    st.markdown("""
    Bước này giúp chuẩn bị dữ liệu đầu vào cho AI: **Lọc ảnh rỗng, loại bỏ ảnh trùng lặp, xóa ảnh lỗi/bị cắt xén**. 
    Sau khi lọc, dữ liệu sẽ được chia thành 2 tập (Train/Validation) và lưu vào các thư mục riêng biệt. 
    **Bạn có thể tạo ra nhiều bộ dữ liệu (như 80/20, 90/10) để so sánh lúc huấn luyện!**
    """)
    st.divider()
    
    parameters = get_default_parameters()
    
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("⚙️ Cấu hình Chia tỷ lệ")
            train_ratio = st.slider("Tỷ lệ tập Huấn luyện (Train) %", 50, 95, 80, help="Phần còn lại sẽ được đưa vào tập Validation (đánh giá).") / 100.0
            val_ratio = 1.0 - train_ratio
            train_pct = int(round(train_ratio * 100))
            val_pct = int(round(val_ratio * 100))
            st.info(f"👉 Dữ liệu sẽ được chia: **{train_pct}% Train** - **{val_pct}% Validation**")
            
        with col2:
            st.subheader("💾 Cấu hình Lưu trữ")
            dataset_name = st.text_input("Tên bộ dữ liệu sinh ra:", value=f"dataset_{train_pct}_{val_pct}", help="Bạn có thể đặt tên bất kỳ. Ví dụ: 'dataset_sieu_sach_90_10'")
            st.caption(f"Thư mục lưu trữ: `data/processed/{dataset_name}/`")
    
    if "is_preprocessing" not in st.session_state:
        st.session_state["is_preprocessing"] = False
        
    if not st.session_state["is_preprocessing"]:
        st.write("")
        col_btn, _ = st.columns([1, 2])
        if col_btn.button("🚀 Bắt đầu Tiền Xử Lý (Cleaning & Splitting)", type="primary", use_container_width=True, disabled=is_training or is_evaluating):
            if not dataset_name.strip():
                st.error("❌ Tên thư mục không được để trống!")
            else:
                st.session_state["is_preprocessing"] = True
                st.session_state["prep_dataset_name"] = dataset_name.strip()
                st.session_state["prep_train_ratio"] = train_ratio
                st.session_state["prep_val_ratio"] = val_ratio
                st.rerun()
                
    if st.session_state.get("is_preprocessing", False):
        import hashlib
        import random
        
        st.warning("⚠️ Hệ thống đang quét và xử lý dữ liệu. Vui lòng chờ cho đến khi hoàn tất!")
        
        dataset_name_val = st.session_state["prep_dataset_name"]
        train_ratio_val = st.session_state["prep_train_ratio"]
        val_ratio_val = st.session_state["prep_val_ratio"]
        
        cat_dir = parameters.get('path_cats', 'data/PetImages/Cat')
        dog_dir = parameters.get('path_dogs', 'data/PetImages/Dog')
        
        processed_dir = os.path.join("data", "processed", dataset_name_val)
        train_cat = os.path.join(processed_dir, "train", "Cat")
        train_dog = os.path.join(processed_dir, "train", "Dog")
        val_cat = os.path.join(processed_dir, "val", "Cat")
        val_dog = os.path.join(processed_dir, "val", "Dog")
        
        # Xoá cũ nếu đã có
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
            
        os.makedirs(train_cat, exist_ok=True)
        os.makedirs(train_dog, exist_ok=True)
        os.makedirs(val_cat, exist_ok=True)
        os.makedirs(val_dog, exist_ok=True)
        
        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        
        valid_cats = []
        valid_dogs = []
        
        def filter_files(src_dir, valid_list, counters):
            if not os.path.exists(src_dir): return
            files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
            counters['total'] += len(files)
            hashes = set()
            for idx, p in enumerate(files):
                if idx % 100 == 0:
                    progress_text.text(f"Đang quét {src_dir}: {idx}/{len(files)}")
                    progress_bar.progress(idx / len(files))
                
                if os.path.getsize(p) == 0:
                    counters['zero_byte'] += 1
                    continue
                    
                try:
                    with open(p, 'rb') as file:
                        h = hashlib.md5(file.read()).hexdigest()
                        if h in hashes:
                            counters['duplicate'] += 1
                            continue
                        hashes.add(h)
                except: continue
                
                try:
                    with Image.open(p) as img:
                        img.load()
                    valid_list.append(p)
                except:
                    counters['corrupted'] += 1
                    pass
                    
        stats = {'total': 0, 'zero_byte': 0, 'duplicate': 0, 'corrupted': 0}
        filter_files(cat_dir, valid_cats, stats)
        filter_files(dog_dir, valid_dogs, stats)
        
        # Sắp xếp và xáo trộn ngẫu nhiên
        random.seed(42)
        random.shuffle(valid_cats)
        random.shuffle(valid_dogs)
        
        def split_and_copy(files, train_dst, val_dst):
            split_idx = int(round(len(files) * train_ratio_val))
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            
            for idx, f in enumerate(train_files):
                if idx % 100 == 0:
                    progress_text.text(f"Copying to {train_dst}... {idx}/{len(train_files)}")
                shutil.copy2(f, os.path.join(train_dst, os.path.basename(f)))
            for idx, f in enumerate(val_files):
                if idx % 100 == 0:
                    progress_text.text(f"Copying to {val_dst}... {idx}/{len(val_files)}")
                shutil.copy2(f, os.path.join(val_dst, os.path.basename(f)))
            return len(train_files), len(val_files)
            
        progress_text.text("Đang phân bổ dữ liệu Cat...")
        t_c, v_c = split_and_copy(valid_cats, train_cat, val_cat)
        progress_text.text("Đang phân bổ dữ liệu Dog...")
        t_d, v_d = split_and_copy(valid_dogs, train_dog, val_dog)
        
        total_train = t_c + t_d
        total_val = v_c + v_d
        
        # Save metadata
        metadata = {
            "name": dataset_name_val,
            "train_ratio": train_ratio_val,
            "val_ratio": val_ratio_val,
            "total_raw": stats['total'],
            "zero_byte": stats['zero_byte'],
            "duplicate": stats['duplicate'],
            "corrupted": stats['corrupted'],
            "total_train": total_train,
            "total_val": total_val,
            "train_cats": t_c,
            "train_dogs": t_d,
            "val_cats": v_c,
            "val_dogs": v_d,
            "timestamp": time.time()
        }
        with open(os.path.join(processed_dir, "metadata.json"), "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        st.session_state["prep_result_metadata"] = metadata
        st.session_state["prep_result_stats"] = stats
        st.session_state["is_preprocessing"] = False
        st.rerun()
        
    # Hiện kết quả sau khi xử lý xong
    if "prep_result_metadata" in st.session_state and not st.session_state.get("is_preprocessing", False):
        metadata = st.session_state["prep_result_metadata"]
        stats = st.session_state["prep_result_stats"]
        
        st.success(f"🎉 Đã hoàn tất xử lý dữ liệu! Kết quả được lưu tại: `data/processed/{metadata['name']}/`")
        
        # Bảng hiển thị thông tin bằng st.metric
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng ảnh thô", f"{stats['total']:,}")
        col2.metric("Tập Train", f"{metadata['total_train']:,}")
        col3.metric("Tập Validation", f"{metadata['total_val']:,}")
        col4.metric("Loại bỏ (Lỗi/Trùng)", f"{stats['zero_byte'] + stats['duplicate'] + stats['corrupted']:,}")
        
        with st.expander("📊 Xem Chi Tiết Báo Cáo Xóa Rác", expanded=True):
            st.markdown(f"""
            - **Tổng số ảnh quét được:** {stats['total']:,}
            - **Ảnh dung lượng rỗng (0 KB):** {stats['zero_byte']:,} ảnh
            - **Ảnh trùng lặp mã MD5:** {stats['duplicate']:,} ảnh
            - **Ảnh bị lỗi cấu trúc (Corrupted):** {stats['corrupted']:,} ảnh
            ---
            *Số ảnh hợp lệ còn lại được chia vào Train ({metadata['train_ratio']*100:.0f}%) và Validation ({metadata['val_ratio']*100:.0f}%) thành công.*
            """)

# 1. GIAO DIỆN TAB HUẤN LUYỆN
def train_tab():
    st.header("🧠 Huấn luyện mô hình")
    st.markdown("Chọn một bộ dữ liệu đã được **Tiền xử lý**, sau đó cấu hình các tham số và tiến hành huấn luyện các kiến trúc AI.")
    st.divider()
    
    # --- PHẦN HIỂN THỊ THÔNG TIN DATASET ---
    processed_dir_root = os.path.join("data", "processed")
    if not os.path.exists(processed_dir_root) or not os.listdir(processed_dir_root):
        st.error("⚠️ Kho Dữ Liệu Trống!")
        st.info("Chưa tìm thấy tập dữ liệu tiền xử lý nào. Bạn vui lòng sang tab **Tiền xử lý Dữ liệu** để tạo ít nhất 1 bộ dữ liệu trước khi huấn luyện.")
        return
        
    available_datasets = [d for d in os.listdir(processed_dir_root) if os.path.isdir(os.path.join(processed_dir_root, d))]
    selected_dataset = st.selectbox("📂 Chọn bộ dữ liệu để huấn luyện:", available_datasets, help="Mô hình sẽ sử dụng trực tiếp bộ dữ liệu này.")
    
    metadata_path = os.path.join(processed_dir_root, selected_dataset, "metadata.json")
    if not os.path.exists(metadata_path):
        st.error(f"Thư mục {selected_dataset} bị lỗi (thiếu file metadata.json)")
        return
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    with st.expander(f"📊 Chi tiết Bộ Dữ liệu ({selected_dataset})", expanded=True):
        st.markdown(f"""
        **1. Quá trình làm sạch:**
        - **Dữ liệu thô gốc:** `{metadata.get('total_raw', 0):,}` ảnh.
        - **Đã loại bỏ (Rác):** {metadata.get('zero_byte', 0)} ảnh rỗng, {metadata.get('duplicate', 0)} ảnh trùng, {metadata.get('corrupted', 0)} ảnh lỗi.
        
        **2. Nạp vào mạng Neural:**
        - **Tập Train ({metadata['train_ratio']*100:.0f}%):** `{metadata['total_train']:,}` ảnh (Chó: {metadata['train_dogs']:,} | Mèo: {metadata['train_cats']:,})
        - **Tập Validation ({metadata['val_ratio']*100:.0f}%):** `{metadata['total_val']:,}` ảnh (Chó: {metadata['val_dogs']:,} | Mèo: {metadata['val_cats']:,})
        - **Thư mục:** `{os.path.join(processed_dir_root, selected_dataset)}`
        """)
        
    state = _get_train_state_snapshot() # Lấy trạng thái hiện tại từ Thread

    # Nếu Thread báo đã train xong, xóa cache RAM để giao diện tải mô hình mới
    if state["needs_cache_clear"]:
        clear_model_caches("Huấn luyện kết thúc, model cache đã được làm mới.")
        _ack_cache_clear_done()

    # Dropdown menu chọn 1 lúc nhiều model
    selected_model_labels = st.multiselect(
        "Chọn model để train",
        MODEL_LABELS,
        default=[MODEL_LABELS[0]],
    )
    # Ô nhập số
    epochs = st.number_input("Số epoch", min_value=1, max_value=100, value=5)
    # Checkbox
    ignore_checkpoint = st.checkbox("Train lại từ đầu (ignore checkpoint)", value=False)

    if not selected_model_labels:
        st.warning("Vui lòng chọn ít nhất 1 model")

    # Nút bấm chính
    train_clicked = st.button("Train", type="primary", disabled=state["running"] or st.session_state.get("is_preprocessing", False) or is_evaluating)
    if train_clicked and selected_model_labels and not state["running"]:
        # Khởi động luồng (Thread) chạy ngầm phía sau
        selected_model_keys = [MODEL_LABEL_TO_KEY[label] for label in selected_model_labels]
        _start_training_job(selected_model_keys, int(epochs), bool(ignore_checkpoint), selected_dataset)
        st.rerun() # Refresh lại trình duyệt để xem log chạy ngay

    state = _get_train_state_snapshot()
    # Khối giao diện hiển thị trong lúc đang chạy
    if state["running"]:
        model_label = MODEL_KEY_TO_LABEL.get(state["current_model_key"], "Đang chuẩn bị")
        st.info(f"🚀 Đang huấn luyện: **{model_label}** ({state['current_model_index']}/{state['total_models']})")
        
        # Thanh tiến trình Epoch
        if state["target_epoch"] > 0:
            st.caption(f"Tiến độ tổng thể (Epoch {state['current_epoch']} / {state['target_epoch']})")
            prog = min(1.0, state["current_epoch"] / state["target_epoch"])
            st.progress(prog)
            
        # Thanh tiến trình Batch
        batch_total = state.get("batch_total", 0)
        batch_done = state.get("batch_done", 0)
        epoch_started_at = state.get("epoch_started_at", 0)
        if batch_total > 0:
            elapsed_epoch = time.time() - epoch_started_at if epoch_started_at > 0 else 0
            speed = (batch_done / elapsed_epoch) if elapsed_epoch > 0 else 0
            eta = (batch_total - batch_done) / speed if speed > 0 else 0
            
            # Format thời gian
            eta_m, eta_s = divmod(int(eta), 60)
            elapsed_m, elapsed_s = divmod(int(elapsed_epoch), 60)
            
            st.caption(f"Tiến độ dữ liệu (Batch {batch_done} / {batch_total}) ⏳ Đã chạy: {elapsed_m:02d}:{elapsed_s:02d} | Ước tính còn: {eta_m:02d}:{eta_s:02d} | ⚡ Tốc độ: {speed:.1f} ảnh/s")
            batch_prog = min(1.0, batch_done / batch_total)
            st.progress(batch_prog)
            
        # Hiển thị số liệu (Metrics)
        col1, col2, col3 = st.columns(3)
        col1.metric("Train Loss", f"{state.get('train_loss', 0.0):.4f}")
        col2.metric("Val Loss", f"{state.get('val_loss', 0.0):.4f}")
        col3.metric("Val Accuracy", f"{state.get('val_acc', 0.0):.2%}")

        st.markdown("---")
        # Nút nhấn Dừng hoặc Làm mới
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🛑 Dừng khẩn cấp (Toàn bộ)", type="secondary"):
                _request_stop_training() # Gửi cờ dừng
                st.warning("Đã gửi yêu cầu dừng. Hệ thống sẽ kết thúc an toàn.")
        with col2:
            if st.button("⏭️ Bỏ qua Model hiện tại", type="primary"):
                _request_skip_model()
                st.warning("Đang chờ vòng lặp thoát để chuyển sang Model tiếp theo...")
        with col3:
            st.caption("Giao diện sẽ tự động làm mới mỗi giây.")
            
        # Vòng lặp refresh UI tự động (Trick của Streamlit)
        time.sleep(1.0)
        st.rerun()

    # Cảnh báo sau khi chạy xong
    if state["done"] and not state["running"]:
        if state["error"]:
            st.error(f"Huấn luyện kết thúc với lỗi: {state['error']}")
        elif state["stopped"]:
            st.warning(f"Huấn luyện đã dừng theo yêu cầu. Thời gian đã chạy: {state['elapsed']:.2f}s")
        else:
            st.success(f"Huấn luyện hoàn tất. Tổng thời gian: {state['elapsed']:.2f}s")

    # Khung text để hiển thị list string logs liên tục
    if state["logs"]:
        st.subheader("Nhật ký huấn luyện")
        st.text_area("Logs", value="\n".join(state["logs"]), height=220)


# 2. GIAO DIỆN TAB DỰ ĐOÁN & TRỰC QUAN HÓA
def predict_tab():
    st.header(":mag_right: Phân loại & Trực quan hóa")
    selected_model_labels = st.multiselect(
        "Chọn model để phân loại (nhiều model)",
        MODEL_LABELS,
        default=[MODEL_LABELS[0]],
    )
    if st.button("🔄 Reload models", key="reload_predict_models"):
        clear_model_caches("Models reloaded")

    # Chọn Dataset
    processed_dir_root = os.path.join("data", "processed")
    available_datasets = [d for d in os.listdir(processed_dir_root) if os.path.isdir(os.path.join(processed_dir_root, d))] if os.path.exists(processed_dir_root) else []
    
    if not available_datasets:
        st.warning("⚠️ Chưa có bộ dữ liệu nào. Vui lòng tạo bộ dữ liệu trong tab Tiền xử lý.")
        return
        
    selected_dataset = st.selectbox("📂 Chọn bộ dữ liệu (Model đã train trên dataset này):", available_datasets, key="predict_dataset")

    # Cấu hình Heatmap
    heatmap_weight = st.slider("Độ đậm heatmap (Grad-CAM)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    original_weight = 1.0 - heatmap_weight

    # Nút upload file ảnh từ máy tính user
    uploaded_file = st.file_uploader("Upload ảnh để phân loại", type=["jpg", "jpeg", "png"])
    # Hoặc ô dropbox chọn file đã lưu sẵn
    test_images = os.listdir(TEST_IMAGES_DIR) if os.path.exists(TEST_IMAGES_DIR) else []
    selected_image = st.selectbox("Hoặc chọn ảnh từ test_images", [None] + test_images)
    
    # Hiển thị ảnh review
    img = None
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
    elif selected_image:
        img_path = os.path.join(TEST_IMAGES_DIR, selected_image)
        img = Image.open(img_path).convert("RGB")

    if not selected_model_labels:
        st.info("Vui lòng chọn ít nhất 1 model để dự đoán.")

    # Logic tiến hành dự đoán
    if img is not None and selected_model_labels and st.button("Dự đoán & Trực quan hóa"):
        with st.spinner("Đang khởi tạo các module trực quan hóa..."): 
            os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
            temp_name = "_temp_predict.jpg"
            temp_path = os.path.join(TEST_IMAGES_DIR, temp_name)
            img.save(temp_path)
            
            parameters = get_default_parameters()
            parameters['dataset_name'] = selected_dataset
            selected_model_keys = [MODEL_LABEL_TO_KEY[label] for label in selected_model_labels]
            ensure_model_cache_fresh(selected_model_keys, parameters, "predict_model_signature")
            
        # KHỞI TẠO THANH TIẾN TRÌNH
        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        total_models = len(selected_model_keys)

        try:
            predictor = get_predictor(parameters)
            heatmap_generator = get_heatmap_generator()
            rows = []

            for idx, model_key in enumerate(selected_model_keys):
                model_name = MODEL_KEY_TO_LABEL[model_key]
                progress_text.markdown(f"**⏳ Đang dự đoán & sinh bản đồ nhiệt:** `{model_name}` ({idx+1}/{total_models})...")
                try:
                    wrapped_model = load_model_cached(model_key, parameters)
                    
                    # Dự đoán nhãn
                    pred = predictor.predict(model_key, temp_path)
                    
                    # Sinh Grad-CAM heatmap
                    _, overlay_img, _, _ = heatmap_generator.generate(
                        wrapped_model,
                        temp_path,
                        input_size=int(parameters.get('size', 50)),
                        device=str(parameters.get('device', 'cpu')),
                        original_weight=original_weight,
                        heatmap_weight=heatmap_weight,
                    )
                    
                    rows.append({
                        "Model": model_name,
                        "Prediction": pred.label.upper(),
                        "Confidence": f"{pred.confidence:.2%}",
                        "Heatmap": overlay_img,
                        "_confidence": pred.confidence,
                    })
                except Exception as model_err:
                    rows.append({
                        "Model": model_name,
                        "Prediction": f"Lỗi: {model_err}",
                        "Confidence": "-",
                        "Heatmap": None,
                        "_confidence": -1.0,
                    })

                # Cập nhật thanh tiến trình sau mỗi model hoàn tất
                progress_bar.progress((idx + 1) / total_models)

            # Xóa thanh tiến trình sau khi hoàn tất
            progress_text.empty()
            progress_bar.empty()

            # Sắp xếp model tốt nhất
            rows = sorted(rows, key=lambda x: x["_confidence"], reverse=True)
            best_row = next((row for row in rows if row["_confidence"] >= 0), None)

            if best_row is not None:
                st.success(f"✅ Model tốt nhất: {best_row['Model']} | {best_row['Prediction']} | {best_row['Confidence']}")

            st.markdown("### Chi tiết dự đoán và Grad-CAM")
            
            for idx, row in enumerate(rows):
                st.markdown(f"**{row['Model']}** -> Dự đoán: `{row['Prediction']}` (Tự tin: {row['Confidence']})")
                cols = st.columns(2)
                cols[0].image(img, caption="Ảnh gốc", use_container_width=True)
                if row.get("Heatmap") is not None:
                    cols[1].image(row["Heatmap"], caption=f"Grad-CAM ({row['Model']})", use_container_width=True)
                else:
                    cols[1].warning("Không sinh được Heatmap")
                st.markdown("---")

        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# TAB BIỂU ĐỒ LỊCH SỬ HUẤN LUYỆN
def chart_tab():
    st.header("📈 Biểu đồ Lịch sử Huấn luyện")
    selected_model_labels = st.multiselect(
        "Chọn các model để vẽ biểu đồ so sánh",
        MODEL_LABELS,
        default=MODEL_LABELS,
    )
    
    processed_dir_root = os.path.join("data", "processed")
    available_datasets = [d for d in os.listdir(processed_dir_root) if os.path.isdir(os.path.join(processed_dir_root, d))] if os.path.exists(processed_dir_root) else []
    
    if not available_datasets:
        st.warning("⚠️ Chưa có bộ dữ liệu nào.")
        return
        
    selected_dataset = st.selectbox("📂 Chọn bộ dữ liệu để xem lịch sử:", available_datasets, key="chart_dataset")
    
    if not selected_model_labels:
        st.info("Vui lòng chọn ít nhất 1 model.")
        return
        
    import json
    import pandas as pd
    
    all_data = []
    for label in selected_model_labels:
        model_key = MODEL_LABEL_TO_KEY[label]
        history_path = os.path.join("checkpoints", selected_dataset, model_key, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                for item in history:
                    item['Model'] = label
                    all_data.append(item)
            except Exception as e:
                st.warning(f"Không thể đọc lịch sử của {label}: {e}")
                
    if not all_data:
        st.warning("Chưa có dữ liệu lịch sử huấn luyện (history.json) cho các model đã chọn. Vui lòng Huấn luyện model trước!")
        return
        
    # Chuyển đổi thành DataFrame để vẽ biểu đồ dễ dàng
    df = pd.DataFrame(all_data)
    # Loại bỏ các epoch bị trùng lặp do việc train lại mà không xoá lịch sử cũ
    df = df.drop_duplicates(subset=['epoch', 'Model'], keep='last')
    
    if 'epoch' not in df.columns:
        st.error("Dữ liệu lịch sử không hợp lệ (thiếu cột 'epoch').")
        return
        
    st.markdown("### So sánh Mất mát (Loss)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Train Loss**")
        if 'train_loss' in df.columns:
            train_loss_df = df.pivot(index='epoch', columns='Model', values='train_loss')
            st.line_chart(train_loss_df)
        else:
            st.info("Không có dữ liệu Train Loss")
            
    with col2:
        st.markdown("**Validation Loss**")
        if 'val_loss' in df.columns:
            val_loss_df = df.pivot(index='epoch', columns='Model', values='val_loss')
            st.line_chart(val_loss_df)
        else:
            st.info("Không có dữ liệu Validation Loss")
            
    st.markdown("### So sánh Độ chính xác (Validation Accuracy)")
    if 'val_acc' in df.columns:
        val_acc_df = df.pivot(index='epoch', columns='Model', values='val_acc')
        st.line_chart(val_acc_df)
    else:
        st.info("Không có dữ liệu Validation Accuracy")
        
    st.markdown("### Chi tiết dữ liệu dạng Bảng")
    st.dataframe(df, use_container_width=True)

# 3. GIAO DIỆN QUẢN LÝ DỮ LIỆU TEST
def manage_data_tab():
    st.header(":file_folder: Quản lý Ảnh Dự đoán (Test Images)")
    st.markdown("---")
    
    # --- QUẢN LÝ DỮ LIỆU TEST ---
    st.subheader("2. Quản lý Ảnh dự đoán (Test Images)")
    col1, col2 = st.columns(2)
    with col1:
        # Nút chuyển tab mở trang google image
        if st.button("Tải ảnh từ Google"):
            q = st.text_input("Nhập từ khoá tìm kiếm", value="cat")
            if q:
                import webbrowser
                webbrowser.open(f"https://www.google.com/search?tbm=isch&q={q}")
    with col2:
        # Nút xóa trắng thư mục
        if st.button("Xóa toàn bộ ảnh trong test_images", type="primary"):
            if os.path.exists(TEST_IMAGES_DIR):
                for f in os.listdir(TEST_IMAGES_DIR):
                    try:
                        os.remove(os.path.join(TEST_IMAGES_DIR, f))
                    except Exception:
                        pass
                st.success("Đã xóa toàn bộ ảnh trong test_images!")
    
    st.subheader("Danh sách ảnh trong test_images:")
    # Render toàn bộ ảnh đang lưu để user xem
    if os.path.exists(TEST_IMAGES_DIR):
        files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))]
        if files:
            for f in files:
                try:
                    st.image(os.path.join(TEST_IMAGES_DIR, f), width=120, caption=f)
                except Exception as e:
                    st.warning(f"Không thể mở ảnh {f}: {e}")
        else:
            st.info("Thư mục test_images chưa có ảnh hợp lệ.")
    else:
        st.info("Thư mục test_images chưa có ảnh.")

# 4. GIAO DIỆN TAB ĐÁNH GIÁ (EVALUATE)
def evaluate_tab():
    st.header(":bar_chart: Đánh giá model trên validation set")
    
    if "is_evaluating" not in st.session_state:
        st.session_state["is_evaluating"] = False
        
    if not st.session_state["is_evaluating"]:
        selected_model_labels = st.multiselect(
            "Chọn model để đánh giá (nhiều model)",
            MODEL_LABELS,
            default=[MODEL_LABELS[0]],
            key="eval_models_select",
        )
        if st.button("🔄 Reload models", key="reload_eval_models"):
            clear_model_caches("Models reloaded")
    
        # --- PHẦN CHỌN BỘ DỮ LIỆU ĐỂ ĐÁNH GIÁ ---
        processed_dir_root = os.path.join("data", "processed")
        if not os.path.exists(processed_dir_root) or not os.listdir(processed_dir_root):
            st.warning("⚠️ Chưa tìm thấy bộ dữ liệu tiền xử lý nào để đánh giá.")
            return
            
        available_datasets = [d for d in os.listdir(processed_dir_root) if os.path.isdir(os.path.join(processed_dir_root, d))]
        selected_dataset = st.selectbox("📂 Chọn bộ dữ liệu validation để đánh giá:", available_datasets)
    
        if not selected_model_labels:
            st.info("Vui lòng chọn ít nhất 1 model để đánh giá.")
    
        if selected_model_labels and st.button("Evaluate"):
            st.session_state["is_evaluating"] = True
            st.session_state["eval_selected_models"] = selected_model_labels
            st.session_state["eval_selected_dataset"] = selected_dataset
            st.rerun()
            
    if st.session_state["is_evaluating"]:
        st.warning("⚠️ Đang thực hiện đánh giá. Vui lòng KHÔNG chuyển tab hoặc tải lại trang!")
        selected_model_labels = st.session_state["eval_selected_models"]
        selected_dataset = st.session_state["eval_selected_dataset"]
        
        with st.spinner("Đang chuẩn bị dữ liệu đánh giá (chỉ mất vài giây)..."):
            try:
                # Lấy thư viện tính toán chuẩn của Machine Learning
                sklearn_metrics = importlib.import_module("sklearn.metrics")
                precision_score = sklearn_metrics.precision_score
                recall_score = sklearn_metrics.recall_score
                f1_score = sklearn_metrics.f1_score
            except Exception as e:
                st.error(f"Thiếu sklearn để tính metric mở rộng: {e}")
                st.info("Cài đặt bằng lệnh: pip install scikit-learn")
                st.session_state["is_evaluating"] = False
                return

            parameters = get_default_parameters()
            parameters['dataset_name'] = selected_dataset # Truyền tên dataset vào params
            _, val = load_data_catsVsdogs(parameters) # Gọi hàm sinh dataset để lấy tập valid
            selected_model_keys = [MODEL_LABEL_TO_KEY[label] for label in selected_model_labels]
            ensure_model_cache_fresh(selected_model_keys, parameters, "eval_model_signature")

        # KHỞI TẠO THANH TIẾN TRÌNH
        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        total_models = len(selected_model_keys)

        rows = []
        for idx, model_key in enumerate(selected_model_keys):
            model_name = MODEL_KEY_TO_LABEL[model_key]
            progress_text.markdown(f"**⏳ Đang tiến hành chạy đánh giá:** `{model_name}` (Mô hình {idx+1}/{total_models})... Việc này có thể mất vài phút.")
            
            try:
                start_time = time.time()
                trainer_obj = load_eval_model_cached(model_key, parameters, val)
                trainer_obj.model.eval()

                # Đánh giá cơ bản lấy tỷ lệ đúng (Accuracy)
                _, val_acc = trainer_obj.evaluate(val)
                # Lấy nhãn để tính thêm Precision, Recall, F1
                y_true, y_pred = collect_predictions(
                    model_key,
                    trainer_obj,
                    val,
                    input_size=int(parameters.get("size", 50)),
                )

                precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
                recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
                f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
                elapsed = time.time() - start_time
                total_params = count_params(trainer_obj.model)

                rows.append({
                    "Model": model_name,
                    "Accuracy": f"{val_acc:.2%}",
                    "Precision": f"{precision:.2%}",
                    "Recall": f"{recall:.2%}",
                    "F1": f"{f1:.2%}",
                    "Params": f"{total_params:,}",
                    "Time (s)": f"{elapsed:.3f}",
                    "_f1": f1,
                })
            except Exception as model_err:
                rows.append({
                    "Model": model_name,
                    "Accuracy": "-",
                    "Precision": "-",
                    "Recall": "-",
                    "F1": "-",
                    "Params": "-",
                    "Time (s)": "-",
                    "_f1": -1.0,
                })
                st.error(f"{model_name} evaluate lỗi: {model_err}")

            # Cập nhật thanh tiến trình sau mỗi model
            progress_bar.progress((idx + 1) / total_models)

        # Kết thúc vòng lặp, xoá thanh tiến trình đi
        progress_text.empty()
        progress_bar.empty()

        # Sắp xếp xếp hạng F1
        rows = sorted(rows, key=lambda x: x["_f1"], reverse=True)
        best_row = next((row for row in rows if row["_f1"] >= 0), None)
        
        # Render thành dataframe
        table_rows = [
            {
                "Model": row["Model"],
                "Accuracy": row["Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "Params": row["Params"],
                "Time": row["Time (s)"],
            }
            for row in rows
        ]
        
        st.session_state["eval_result_best"] = best_row
        st.session_state["eval_result_table"] = table_rows
        st.session_state["is_evaluating"] = False
        st.rerun()

    if "eval_result_table" in st.session_state and not st.session_state.get("is_evaluating", False):
        best_row = st.session_state["eval_result_best"]
        if best_row is not None:
            st.success(f"Model tốt nhất theo F1: {best_row['Model']} ({best_row['F1']})")
        st.dataframe(st.session_state["eval_result_table"], use_container_width=True, hide_index=True)

# ================= ĐIỂM CHẠY CHÍNH (ENTRY POINT) =================
if __name__ == "__main__":
    # Router chuyển trang theo kết quả lựa chọn menu ở sidebar
    if page == "Tiền xử lý Dữ liệu":
        preprocess_tab()
    elif page == "Huấn luyện":
        train_tab()
    elif page == "Biểu đồ Lịch sử":
        chart_tab()
    elif page == "Dự đoán & Phân tích":
        predict_tab()
    elif page == "Quản lý dữ liệu":
        manage_data_tab()
    elif page == "Đánh giá":
        evaluate_tab()

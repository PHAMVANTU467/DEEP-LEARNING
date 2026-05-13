import os
import sys

if __name__ == "__main__":
    import subprocess
    # Đảm bảo Python nhận diện được thư mục gốc chứa `src`
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{root_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = root_dir
        
    # Chạy giao diện Streamlit bằng lệnh system
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app/main.py"], env=env)

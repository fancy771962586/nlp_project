import subprocess
import sys
import time

# 定义要运行的 Python 文件
FRONTEND_FILE = 'frontend.py'
BACKEND_FILE = 'backend.py'

# 启动前端和后端进程
frontend_process = subprocess.Popen([sys.executable, FRONTEND_FILE])
backend_process = subprocess.Popen([sys.executable, BACKEND_FILE])

# 存储进程对象，以便稍后可以使用它们来终止进程
processes = {
    'frontend': frontend_process,
    'backend': backend_process
}

if __name__ == '__main__':
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating processes...")
        for name, process in processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)  # 等待进程终止或超时
                print(f"{name} process termination attempted.")
            except Exception as e:
                print(f"Error terminating {name} process: {e}")
        print("All termination attempts completed. Exiting gracefully.")
        sys.exit(0)
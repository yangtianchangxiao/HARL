import subprocess
import time

# 运行fuser命令获取所有进程ID
cmd = "fuser -v /dev/nvidia2"
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
stdout = stdout.decode().strip()

# 提取进程ID
pids = [pid for pid in stdout.split() if pid.isdigit()]

# 逐个杀死进程
for pid in pids:
    try:
        print(f"Killing process {pid}...")
        kill_cmd = f"kill -9 {pid}"
        subprocess.run(kill_cmd, shell=True, check=True)
        print(f"Process {pid} killed.")
        time.sleep(0.1)  # 可以调整这个时间，如果需要的话
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill process {pid}: {e}")

print("All processes killed.")

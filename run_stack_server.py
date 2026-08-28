import subprocess
import os
import time
import urllib.request
import json
import sys
import psutil

sys.stdout.reconfigure(encoding='utf-8')

LLAMA_BIN = r"C:\Users\admin\source\llama.cpp\build\bin\Release\llama-server.exe"
MODEL_PATH = r"C:\Users\admin\gemma4-turbo-family\haven-chat-v5.0.gguf"
HAVEN_SERVER_BIN = r"C:\Users\admin\source\haven-cpp\build\haven-server.exe"
if not os.path.exists(HAVEN_SERVER_BIN):
    HAVEN_SERVER_BIN = r"C:\Users\admin\source\haven-cpp\haven-server.exe"

SD_SERVER_BIN = r"C:\Users\admin\stable-diffusion-cpp\sd-server.exe"
SD_MODEL_PATH = r"C:\Users\admin\stable-diffusion-cpp\models\DreamShaper8_LCM_q4_0.gguf"
TAESD_PATH = r"C:\Users\admin\stable-diffusion-cpp\models\taesd.safetensors"

STANDALONE_MODE = False # Use stable hybrid backend for live conversations while haven-cpp forward pass is refined

# 1. Clean up old processes
for p in psutil.process_iter(['pid', 'name']):
    try:
        pname = p.info['name'].lower()
        if any(x in pname for x in ['llama-server', 'haven-server', 'sd-server', 'debug_gen', 'test-standalone', 'test-logits']):
            p.kill()
    except Exception:
        pass

time.sleep(1)

# 2. Launch llama-server only if NOT in standalone mode
llama_proc = None
if not STANDALONE_MODE:
    llama_cmd = [
        LLAMA_BIN,
        "-m", MODEL_PATH,
        "--port", "11436",
        "--host", "127.0.0.1",
        "-c", "8192",
        "--threads", "8"
    ]
    print("Starting llama-server (Gemma 4 Gated Delta Net Backend) on port 11436...")
    llama_proc = subprocess.Popen(llama_cmd)
else:
    print("🚀 STANDALONE SOVEREIGN MODE: 100% Native bare-metal haven-cpp engine active (llama-server disabled)!")

# 3. Launch sd-server (Stable Diffusion C++) on port 8085 with fast 6-step LCM
sd_cmd = [
    SD_SERVER_BIN,
    "-m", SD_MODEL_PATH,
    "--taesd", TAESD_PATH,
    "--sampling-method", "lcm",
    "--steps", "6",
    "--cfg-scale", "1.8",
    "--lora-model-dir", r"C:\Users\admin\stable-diffusion-cpp\models\lora-models",
    "--embd-dir", r"C:\Users\admin\stable-diffusion-cpp\models\embeddings",
    "--listen-ip", "0.0.0.0",
    "--listen-port", "8085",
    "--threads", "8"
]
print("Starting sd-server (Stable Diffusion C++) on port 8085 with 6-step LCM...")
sd_proc = subprocess.Popen(sd_cmd)

# 4. Launch haven-server Sovereign C++ Engine & Web Studio on port 11438
print("Starting haven-server Sovereign Master Micro-Server on port 11438...")
haven_proc = subprocess.Popen([HAVEN_SERVER_BIN, "11438"], cwd=r"C:\Users\admin\source\haven-cpp")

# 5. Wait for all 3 services to be healthy
ready_llama = False
for _ in range(40):
    time.sleep(0.5)
    try:
        req = urllib.request.Request("http://127.0.0.1:11436/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            ready_llama = True
            break
    except Exception:
        pass

ready_haven = False
for _ in range(30):
    time.sleep(0.5)
    try:
        req = urllib.request.Request("http://127.0.0.1:11438/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            ready_haven = True
            break
    except Exception:
        pass

ready_sd = False
for _ in range(20):
    time.sleep(0.5)
    try:
        req = urllib.request.Request("http://127.0.0.1:8085/")
        with urllib.request.urlopen(req, timeout=1) as resp:
            ready_sd = True
            break
    except Exception:
        pass

print(f"\n==================================================================")
print(f"🏛️ HAVEN SOVEREIGN ECOSYSTEM ONLINE")
print(f"==================================================================")
print(f"  ✓ llama-server (Port 11436): {'ONLINE (Gemma 4 Gated Delta Net)' if ready_llama else 'STARTING'}")
print(f"  ✓ haven-server (Port 11438): {'ONLINE (Sovereign Web Studio & Memory Vault)' if ready_haven else 'STARTING'}")
print(f"  ✓ sd-server    (Port 8085):  {'ONLINE (DreamShaper 6-Step LCM)' if ready_sd else 'STARTING'}")

# Print memory stats
procs = []
if llama_proc: procs.append((llama_proc.pid, "llama-server (Gemma 4)"))
if haven_proc: procs.append((haven_proc.pid, "haven-server (Sovereign Studio)"))
if sd_proc: procs.append((sd_proc.pid, "sd-server (Diffusion)"))

for pid, name in procs:
    try:
        proc = psutil.Process(pid)
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        print(f"  ✓ {name} (PID {pid}): {rss_mb:.2f} MB RAM")
    except Exception as e:
        print(f"  {name} stat error:", e)

print("\n🚀 Endpoints Available:")
print("   - Web UI Studio:             http://localhost:11438/")
print("   - Cognitive Memory Vault:    http://localhost:11438/memories")
print("   - Visual Uploads Gallery:    http://localhost:11438/uploads/")
print("   - Sovereign Chat API:        http://localhost:11438/v1/chat/completions")
print("   - Memory API:                http://localhost:11438/api/memories")
print("   - Image Generations API:     http://localhost:11438/v1/images/generations")
print("   - Models API:                http://localhost:11438/v1/models\n")

try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    llama_proc.terminate()
    haven_proc.terminate()
    sd_proc.terminate()

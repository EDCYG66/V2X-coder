import os

# 在导入 TensorFlow 之前兜底设置 WSL 的 CUDA 用户态库路径
if '/usr/lib/wsl/lib' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
# 建议：开启显存按需增长（也可在 TF 配置里设置）
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print("Set memory growth failed:", e)
    return gpus

def check_tf_gpu():
    print("TF version:", tf.__version__)
    print("TF path:", tf.__path__)
    print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', ''))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)'))

    # 构建信息（pip 安装的 TF 2.12 通常是 CUDA 11.8 + cuDNN 8.6）
    try:
        info = tf.sysconfig.get_build_info()
        print("Built with CUDA?", info.get("is_cuda_build"))
        print("CUDA version:", info.get("cuda_version"))
        print("cuDNN version:", info.get("cudnn_version"))
    except Exception as e:
        print("Build info unavailable:", e)

    physical = tf.config.list_physical_devices('GPU')
    logical = tf.config.list_logical_devices('GPU')
    print("Physical GPUs:", physical)
    print("Logical GPUs:", logical)

    # 做一次计算看实际设备
    a = tf.random.uniform((1024, 1024))
    b = tf.random.uniform((1024, 1024))
    c = tf.matmul(a, b)
    print("Sample matmul device:", c.device)

    # 尝试显式放到 GPU:0（如不可用会抛异常）
    try:
        with tf.device('/GPU:0'):
            x = tf.matmul(a, b)
        print("Explicit /GPU:0 matmul device:", x.device)
    except Exception as e:
        print("Explicit /GPU:0 failed:", e)

if __name__ == "__main__":
    setup_gpu()
    check_tf_gpu()

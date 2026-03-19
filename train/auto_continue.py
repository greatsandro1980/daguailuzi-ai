"""
自动监控 PPO 训练完成，然后接续 45 万局纯自我对弈。
用法：python3 auto_continue.py  （后台运行即可）
"""
import os
import time
import subprocess
import sys

TRAIN_DIR   = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(TRAIN_DIR, 'train_ppo.log')
LOG_CONT    = os.path.join(TRAIN_DIR, 'train_ppo_continue.log')
LATEST_CKPT = os.path.join(TRAIN_DIR, 'checkpoints_ppo', 'ppo_latest.pt')
CONTINUE_EP = 450000   # 接续训练局数

DONE_MARKER = 'PPO 训练完成'   # train_ppo.py 打印的结束标志


def tail_log(path, n=5):
    """读取日志最后 n 行"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception:
        return ''


def wait_for_done():
    print(f"[auto_continue] 监控中: {LOG_FILE}")
    print(f"[auto_continue] 等待训练完成标志: '{DONE_MARKER}'")
    while True:
        text = tail_log(LOG_FILE, 10)
        if DONE_MARKER in text:
            print("[auto_continue] 检测到训练完成！")
            return
        time.sleep(30)   # 每30秒检查一次


def start_continue():
    if not os.path.exists(LATEST_CKPT):
        print(f"[auto_continue] ❌ 找不到 checkpoint: {LATEST_CKPT}")
        sys.exit(1)

    print(f"[auto_continue] 启动接续训练：{CONTINUE_EP} 局纯自我对弈")
    print(f"[auto_continue] 日志输出到: {LOG_CONT}")

    cmd = [
        sys.executable, '-u', os.path.join(TRAIN_DIR, 'train_ppo.py'),
        '--resume',   LATEST_CKPT,
        '--episodes', str(CONTINUE_EP),
        '--stage',    'self',   # 强制纯自我对弈
    ]

    with open(LOG_CONT, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f,
                                cwd=TRAIN_DIR)
    print(f"[auto_continue] 接续训练已启动 PID={proc.pid}")
    return proc


if __name__ == '__main__':
    wait_for_done()
    time.sleep(5)   # 等 checkpoint 写完
    proc = start_continue()
    print(f"[auto_continue] 监控退出，接续训练 PID={proc.pid} 正在运行")
    print(f"[auto_continue] 查看进度: tail -f {LOG_CONT}")

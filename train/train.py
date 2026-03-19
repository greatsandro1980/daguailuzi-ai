"""
大怪路子强化学习训练脚本
算法：Actor-Critic + 自我对弈
M1 Mac 使用 MPS 加速
"""
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from game_env import DaguaiEnv, encode_state
from model import DaguaiNet, ReplayBuffer, select_action

# ─── 超参数 ──────────────────────────────────────────
CFG = {
    'lr':            3e-4,
    'gamma':         0.99,        # 折扣因子
    'entropy_coef':  0.02,        # 熵正则（鼓励探索）
    'value_coef':    0.5,
    'clip_grad':     0.5,
    'batch_episodes':16,          # 每次更新用几局经验
    'temperature':   1.0,         # 初始温度（越高越随机）
    'min_temperature':0.3,
    'save_interval': 500,         # 每N局保存一次
    'log_interval':  100,
    'total_episodes':100000,
    'hidden_dim':    512,
}


def train_one_episode(net, env, device, temperature, episode=0, total=100000):
    """跑一局完整游戏，收集经验
    课程学习：
      ep <  20000: 3个AI + 3个随机对手（席位1,3,5为随机）
      ep <  50000: 5个AI + 1个随机对手（席位5为随机）
      ep >= 50000: 纯自我对弈
    """
    progress = episode / total
    if progress < 0.2:
        random_seats = {1, 3, 5}
    elif progress < 0.5:
        random_seats = {5}
    else:
        random_seats = set()

    obs    = env.reset()
    buffer = ReplayBuffer()
    steps  = 0
    max_steps = 2000  # 防死循环

    while not env.done and steps < max_steps:
        cp = obs['current_player']

        if cp in random_seats:
            # 随机对手：从合法动作里随机选
            action = random.choice(obs['legal_actions'])
            obs, rewards, done, _ = env.step(action)
        else:
            feat   = encode_state(obs)
            action, log_prob, value = select_action(net, obs, device, temperature)
            buffer.add(feat, log_prob, value, cp)
            obs, rewards, done, _ = env.step(action)

        steps += 1

    if env.done:
        buffer.fill_rewards(rewards)

    return buffer, rewards


def compute_loss(buffer, net, device):
    """Actor-Critic 损失"""
    if len(buffer) == 0:
        return None

    states    = torch.FloatTensor(np.array(buffer.states)).to(device)
    log_probs = torch.stack(buffer.log_probs).to(device)
    values    = torch.stack(buffer.values).to(device)
    returns   = torch.FloatTensor(buffer.rewards).to(device)

    # Advantage = R - V
    advantage = returns - values.detach()
    # 归一化 advantage
    if advantage.std() > 1e-8:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # 重新计算价值（用于 value loss）
    _, new_values = net(states)
    new_values = new_values.squeeze(-1)

    # Actor loss (policy gradient)
    actor_loss = -(log_probs * advantage).mean()

    # Critic loss (MSE)
    critic_loss = F.mse_loss(new_values, returns)

    # 熵正则（鼓励探索，这里简化为 log_prob 的负均值）
    entropy_loss = log_probs.mean()   # 越负越好（熵越大）

    total_loss = (actor_loss
                  + CFG['value_coef'] * critic_loss
                  + CFG['entropy_coef'] * entropy_loss)

    return total_loss, actor_loss.item(), critic_loss.item()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='从checkpoint继续训练')
    parser.add_argument('--episodes', type=int, default=CFG['total_episodes'])
    args = parser.parse_args()

    # 设备选择
    # MPS 在部分 PyTorch 版本有 "Placeholder storage not allocated" bug，暂时强制用 CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ 使用 CUDA 加速")
    else:
        device = torch.device('cpu')
        print("✅ 使用 CPU 训练（M1 多核，速度足够）")

    # 模型
    net = DaguaiNet(hidden_dim=CFG['hidden_dim']).to(device)
    optimizer = Adam(net.parameters(), lr=CFG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)

    # 从 checkpoint 恢复
    start_episode = 0
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_episode = ckpt.get('episode', 0)
        print(f"✅ 从 episode {start_episode} 恢复训练")

    # 训练统计
    reward_history = []
    loss_history   = []
    score_history  = []   # 每局实际得分（0~3）
    win_history    = []   # 胜负（1/0）

    env = DaguaiEnv()
    print(f"\n开始训练，共 {args.episodes} 局...\n")

    accumulated_buffers = []
    t0 = time.time()

    for ep in range(start_episode, args.episodes):
        # 温度退火（前期多探索，后期更贪心）
        progress = ep / args.episodes
        temperature = max(
            CFG['min_temperature'],
            CFG['temperature'] * (1 - progress * 0.8)
        )

        # 收集一局经验
        buffer, final_rewards = train_one_episode(net, env, device, temperature, ep, args.episodes)
        accumulated_buffers.append(buffer)

        # 记录统计
        red_reward  = np.mean([final_rewards[s] for s in [0, 2, 4]])
        reward_history.append(red_reward)
        # 从 env 中读取本局得分和胜负
        score_history.append(getattr(env, 'last_score', 0))
        win_history.append(1.0 if getattr(env, 'last_winner_team', -1) == 0 else 0.0)

        # 每 batch_episodes 局更新一次网络
        if (ep + 1) % CFG['batch_episodes'] == 0:
            net.train()
            total_loss_val = 0.0
            n_updates = 0
            optimizer.zero_grad()

            for buf in accumulated_buffers:
                result = compute_loss(buf, net, device)
                if result:
                    loss, al, cl = result
                    loss.backward()
                    total_loss_val += loss.item()
                    n_updates += 1

            if n_updates > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), CFG['clip_grad'])
                optimizer.step()
                scheduler.step()
                loss_history.append(total_loss_val / n_updates)

            accumulated_buffers = []
            net.eval()

        # 日志打印
        if (ep + 1) % CFG['log_interval'] == 0:
            elapsed = time.time() - t0
            avg_reward = np.mean(reward_history[-CFG['log_interval']:])
            avg_score  = np.mean(score_history[-CFG['log_interval']:])
            win_rate   = np.mean(win_history[-CFG['log_interval']:])
            avg_loss   = np.mean(loss_history[-10:]) if loss_history else 0.0
            eps_per_sec = CFG['log_interval'] / elapsed
            t0 = time.time()

            print(f"[ep {ep+1:>6}] "
                  f"均值奖励={avg_reward:+.3f}  "
                  f"平均得分={avg_score:.2f}/3  "
                  f"胜率={win_rate:.1%}  "
                  f"loss={avg_loss:.4f}  "
                  f"temp={temperature:.2f}  "
                  f"速度={eps_per_sec:.1f}局/s")

        # 保存 checkpoint
        if (ep + 1) % CFG['save_interval'] == 0:
            path = os.path.join(save_dir, f'model_ep{ep+1}.pt')
            torch.save({
                'episode':   ep + 1,
                'model':     net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg':       CFG,
            }, path)
            # 同时保存最新
            latest = os.path.join(save_dir, 'model_latest.pt')
            torch.save({
                'episode':   ep + 1,
                'model':     net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg':       CFG,
            }, latest)
            print(f"  💾 已保存: {path}")

    print("\n训练完成！")
    print(f"最终胜率: {np.mean(win_history[-1000:]):.1%}")
    print(f"最终平均得分: {np.mean(score_history[-1000:]):.2f} / 3.0")


if __name__ == '__main__':
    train()

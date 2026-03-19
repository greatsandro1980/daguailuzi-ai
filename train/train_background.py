"""
大怪路子后台训练脚本
目标：20万局训练，胜率>60%
特性：自动保存检查点、学习率衰减、对手多样化
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime
import time

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from game_env_fast_v2 import FastDaguaiEnvV2 as FastDaguaiEnv, recognize_cards, can_beat_cards

# ─── 配置 ─────────────────────────────────────────────
CFG = {
    'max_steps': 300,
    'batch_episodes': 100,     # 批量更新
    'ppo_epochs': 4,
    'lr': 5e-4,                # 初始学习率
    'lr_decay': 0.9995,        # 学习率衰减
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.02,
    'temperature': 1.2,
    'temperature_decay': 0.9998,
    'min_temperature': 0.2,
    'max_actions': 50,
    'save_interval': 5000,     # 每5000局保存
    'log_interval': 500,       # 每500局记录
}


# ─── 策略网络 ────────────────────────────────────────
class PolicyNet(nn.Module):
    """策略网络"""
    def __init__(self, state_dim=224, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.action_head = nn.Linear(hidden_dim // 2, 54)
    
    def forward(self, x):
        feat = self.backbone(x)
        value = self.value_head(feat).squeeze(-1)
        logits = self.action_head(feat)
        return logits, value


# ─── 对手策略 ────────────────────────────────────────
class RandomOpponent:
    """随机对手"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        return random.choice(actions) if actions else np.zeros(54, dtype=np.int8)


class GreedyOpponent:
    """贪心对手：出最大的牌"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        play_actions = [a for a in actions if np.sum(a) > 0]
        if play_actions:
            # 选择牌数最多的
            return max(play_actions, key=lambda x: int(np.sum(x)))
        return actions[0] if actions else np.zeros(54, dtype=np.int8)


class SmartOpponent:
    """智能对手：考虑牌型"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        play_actions = [a for a in actions if np.sum(a) > 0]
        
        if not play_actions:
            return actions[0] if actions else np.zeros(54, dtype=np.int8)
        
        # 评估每个动作
        best_score = -float('inf')
        best_action = play_actions[0]
        
        for action in play_actions:
            score = 0
            n = int(np.sum(action))
            
            # 牌数越多越好
            score += n * 10
            
            # 牌型加分
            ct, cr = recognize_cards(action, obs['trump_rank'])
            if ct:
                score += ct * 5  # 牌型越大越好
            
            # 如果能压过当前牌，加分
            if np.sum(obs['current_played']) > 0:
                if can_beat_cards(obs['current_played'], action, obs['trump_rank']):
                    score += 20
            
            # 随机扰动
            score += random.random() * 5
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action


class OpponentPool:
    """对手池"""
    def __init__(self):
        self.opponents = [
            ('random', RandomOpponent(), 0.2),
            ('greedy', GreedyOpponent(), 0.3),
            ('smart', SmartOpponent(), 0.5),
        ]
    
    def sample(self):
        """随机采样一个对手"""
        r = random.random()
        cumsum = 0
        for name, opp, prob in self.opponents:
            cumsum += prob
            if r < cumsum:
                return name, opp
        return self.opponents[-1][0], self.opponents[-1][1]


# ─── 状态编码 ────────────────────────────────────────
def encode_state(hand, played, current, hand_sizes, trump_rank, my_team):
    """状态编码"""
    feat = np.zeros(224, dtype=np.float32)
    feat[0:54] = hand.astype(np.float32)
    feat[54:108] = played.astype(np.float32) / 3.0
    feat[108:162] = current.astype(np.float32)
    feat[162:216] = (hand > 0).astype(np.float32)
    feat[216:222] = hand_sizes.astype(np.float32) / 27.0
    feat[222] = trump_rank / 14.0
    feat[223] = float(my_team)
    return feat


# ─── 动作选择 ────────────────────────────────────────
def select_action(net, obs, device, temperature=1.0):
    """选择动作"""
    actions = obs['legal_actions']
    if not actions:
        return np.zeros(54, dtype=np.int8), 0.0, 0.0
    
    # 限制动作数
    play_actions = [a for a in actions if np.sum(a) > 0]
    pass_actions = [a for a in actions if np.sum(a) == 0]
    
    if len(play_actions) > CFG['max_actions']:
        # 按牌数排序，保留多样的动作
        play_actions.sort(key=lambda x: int(np.sum(x)), reverse=True)
        # 保留长牌 + 随机短牌
        long = play_actions[:20]
        short = play_actions[20:]
        if short:
            play_actions = long + random.sample(short, min(15, len(short)))
    
    actions = play_actions + pass_actions[:1]
    
    # 编码状态
    state = encode_state(
        obs['hand'], obs['played_all'], obs['current_played'],
        obs['hand_sizes'], obs['trump_rank'], obs['my_team']
    )
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, value = net(state_t)
    
    # 计算分数
    scores = []
    for action in actions:
        n = int(np.sum(action))
        if n > 0:
            action_t = torch.FloatTensor(action).to(device)
            net_score = (action_t * logits).sum().item()
            # 启发式加分
            ct, cr = recognize_cards(action, obs['trump_rank'])
            heuristic = n * 2 + (ct * 3 if ct else 0)
            scores.append(net_score + heuristic)
        else:
            scores.append(-5.0)  # pass
    
    # Softmax采样
    scores_t = torch.FloatTensor(scores)
    probs = F.softmax(scores_t / temperature, dim=0)
    idx = torch.multinomial(probs, 1).item()
    
    return actions[idx], torch.log(probs[idx] + 1e-8).item(), value.item()


# ─── 收集经验 ────────────────────────────────────────
def collect_episode(net, env, device, temperature, opponent_pool):
    """收集一局经验"""
    obs = env.reset()
    experiences = []
    
    # 为蓝队分配对手
    blue_opponents = {1: opponent_pool.sample()[1],
                      3: opponent_pool.sample()[1],
                      5: opponent_pool.sample()[1]}
    
    while not env.done and env.steps < CFG['max_steps']:
        cp = obs['current_player']
        
        if cp % 2 == 0:  # 红队：PPO
            action, log_prob, value = select_action(net, obs, device, temperature)
            state = encode_state(
                obs['hand'], obs['played_all'], obs['current_played'],
                obs['hand_sizes'], obs['trump_rank'], obs['my_team']
            )
            
            obs, rewards, done, _ = env.step(action)
            
            # 中间奖励
            step_reward = 0.02 * int(np.sum(action)) if np.sum(action) > 0 else -0.02
            
            # 牌权奖励
            if np.sum(action) > 0 and np.sum(env.current) > 0:
                ct, _ = recognize_cards(action, obs['trump_rank'])
                if ct:
                    step_reward += 0.03 * ct
            
            if done:
                step_reward += rewards[cp]
            
            experiences.append((state, action.copy(), log_prob, value, step_reward, done))
        else:  # 蓝队：规则对手
            opponent = blue_opponents.get(cp)
            if opponent:
                action = opponent.select_action(obs)
            else:
                actions = obs['legal_actions']
                action = random.choice(actions) if actions else np.zeros(54, dtype=np.int8)
            obs, _, done, _ = env.step(action)
    
    return experiences, rewards if env.done else np.zeros(6)


# ─── PPO更新 ────────────────────────────────────────
def ppo_update(net, optimizer, all_exp, device):
    """PPO更新"""
    if not all_exp:
        return 0.0
    
    # 计算GAE
    states, actions, old_log_probs, values, rewards, dones = zip(*all_exp)
    
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)
    
    # 计算回报和优势
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    last_ret = 0
    last_adv = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + CFG['gamma'] * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + CFG['gamma'] * CFG['gae_lambda'] * (1 - dones[t]) * last_adv
        last_adv = advantages[t]
        
        returns[t] = rewards[t] + CFG['gamma'] * (1 - dones[t]) * last_ret
        last_rt = returns[t]
    
    # 标准化优势
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 转换为tensor
    states_t = torch.FloatTensor(np.array(states)).to(device)
    actions_t = torch.FloatTensor(np.array(actions)).to(device)
    returns_t = torch.FloatTensor(returns).to(device)
    advantages_t = torch.FloatTensor(advantages).to(device)
    
    total_loss = 0.0
    
    for _ in range(CFG['ppo_epochs']):
        logits, values = net(states_t)
        
        # 价值损失
        value_loss = F.mse_loss(values, returns_t)
        
        # 策略损失
        action_scores = (actions_t * logits).sum(dim=1)
        action_norms = actions_t.sum(dim=1).clamp(min=1.0)
        normalized_scores = action_scores / action_norms
        
        # pass动作
        is_pass = (actions_t.sum(dim=1) == 0)
        normalized_scores = torch.where(is_pass, torch.full_like(normalized_scores, -1.0), normalized_scores)
        
        policy_loss = -(normalized_scores * advantages_t).mean()
        
        # 熵奖励
        entropy = -(normalized_scores * torch.exp(normalized_scores.clamp(-10, 10))).mean()
        
        loss = value_loss + 0.5 * policy_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / CFG['ppo_epochs']


# ─── 状态管理 ────────────────────────────────────────
class TrainingState:
    """训练状态"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.status_file = os.path.join(save_dir, 'training_status.json')
        self.log_file = os.path.join(save_dir, 'training_log.txt')
        
        self.episode = 0
        self.win_rate = 0.0
        self.avg_reward = 0.0
        self.speed = 0.0
        self.start_time = datetime.now()
        self.lr = CFG['lr']
        self.temperature = CFG['temperature']
        
        self.reward_history = deque(maxlen=1000)
        self.win_history = deque(maxlen=1000)
    
    def update(self, ep, reward, win, speed, lr, temp):
        self.episode = ep
        self.reward_history.append(reward)
        self.win_history.append(1.0 if win else 0.0)
        self.avg_reward = np.mean(list(self.reward_history)[-100:])
        self.win_rate = np.mean(list(self.win_history)[-100:])
        self.speed = speed
        self.lr = lr
        self.temperature = temp
    
    def save_status(self):
        """保存状态到JSON"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        status = {
            'episode': self.episode,
            'win_rate': round(float(self.win_rate) * 100, 1),
            'avg_reward': round(float(self.avg_reward), 3),
            'speed': round(float(self.speed), 1),
            'elapsed_hours': round(elapsed / 3600, 2),
            'lr': round(float(self.lr), 6),
            'temperature': round(float(self.temperature), 3),
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def log(self, msg):
        """记录日志"""
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# ─── 主训练 ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200000)
    parser.add_argument('--save_dir', type=str, default='/workspace/projects')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    
    # 初始化
    net = PolicyNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG['lr'])
    env = FastDaguaiEnv()
    opponent_pool = OpponentPool()
    state = TrainingState(args.save_dir)
    
    # 加载最新检查点
    checkpoints = [f for f in os.listdir(args.save_dir) if f.startswith('model_ep') and f.endswith('.pt')]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split('ep')[1].split('.')[0]))
        ckpt = torch.load(os.path.join(args.save_dir, latest), map_location='cpu', weights_only=False)
        net.load_state_dict(ckpt['model'])
        state.episode = ckpt.get('episode', 0)
        print(f"加载检查点: {latest}, 从第{state.episode}局继续")
    
    print(f"🚀 开始训练，目标: {args.episodes}局")
    print(f"   设备: {device}")
    print(f"   初始学习率: {CFG['lr']}")
    print(f"   初始温度: {CFG['temperature']}")
    
    experiences = []
    start_time = time.time()
    last_save = state.episode
    
    for ep in range(state.episode, args.episodes):
        # 学习率和温度衰减
        progress = ep / args.episodes
        lr = CFG['lr'] * (CFG['lr_decay'] ** ep)
        temperature = max(CFG['min_temperature'], CFG['temperature'] * (CFG['temperature_decay'] ** ep))
        
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # 收集经验
        exp, rewards = collect_episode(net, env, device, temperature, opponent_pool)
        experiences.extend(exp)
        
        red_reward = np.mean([rewards[s] for s in [0, 2, 4]])
        red_win = red_reward > 0
        
        state.reward_history.append(red_reward)
        state.win_history.append(1.0 if red_win else 0.0)
        
        # 批量更新
        if len(experiences) >= CFG['batch_episodes'] * 50:  # 约50局经验
            loss = ppo_update(net, optimizer, experiences, device)
            experiences = []
        
        # 更新状态
        elapsed = time.time() - start_time
        if elapsed > 0:
            speed = (ep + 1 - state.episode) / elapsed
        else:
            speed = 0
        state.update(ep + 1, red_reward, red_win, speed, lr, temperature)
        
        # 每次都保存状态文件
        state.save_status()
        
        # 定期保存
        if (ep + 1) % CFG['save_interval'] == 0:
            save_path = os.path.join(args.save_dir, f'model_ep{ep+1}.pt')
            torch.save({
                'model': net.state_dict(),
                'episode': ep + 1,
                'win_rate': state.win_rate,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            state.log(f"保存检查点: {save_path}, 胜率: {state.win_rate:.1%}")
            last_save = ep + 1
        
        # 更新状态文件
        if (ep + 1) % CFG['log_interval'] == 0:
            state.save_status()
            msg = f"Ep {ep+1}/{args.episodes} | 胜率: {state.win_rate:.1%} | 奖励: {state.avg_reward:.3f} | 速度: {state.speed:.1f}/s"
            print(msg)
            state.log(msg)
    
    # 最终保存
    final_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'model': net.state_dict(),
        'episode': args.episodes,
        'win_rate': state.win_rate,
    }, final_path)
    
    elapsed = time.time() - start_time
    print(f"\n✅ 训练完成!")
    print(f"   总局数: {args.episodes}")
    print(f"   总耗时: {elapsed/3600:.1f} 小时")
    print(f"   最终胜率: {state.win_rate:.1%}")
    print(f"   模型文件: {final_path}")


if __name__ == '__main__':
    main()

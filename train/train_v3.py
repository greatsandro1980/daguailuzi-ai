"""
大怪路子训练脚本 v3 - 简化版
策略：学习模仿强规则Bot + 自博弈
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from game_env_fast_v2 import FastDaguaiEnvV2, recognize_cards

# ─── 配置 ─────────────────────────────────────────────
CFG = {
    'max_steps': 300,
    'lr': 1e-3,
    'gamma': 0.99,
    'max_episodes': 200000,
    'save_interval': 5000,
    'batch_size': 256,
}


# ─── 策略网络 ────────────────────────────────────────
class PolicyNet(nn.Module):
    """简单的策略网络：输入状态+动作，输出分数"""
    def __init__(self, state_dim=224, action_dim=54):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


# ─── 状态编码 ────────────────────────────────────────
def encode_state(hand, played, current, hand_sizes, trump_rank, my_team):
    feat = np.zeros(224, dtype=np.float32)
    feat[0:54] = hand.astype(np.float32)
    feat[54:108] = played.astype(np.float32) / 3.0
    feat[108:162] = current.astype(np.float32)
    feat[162:216] = (hand > 0).astype(np.float32)
    feat[216:222] = hand_sizes.astype(np.float32) / 27.0
    feat[222] = trump_rank / 14.0
    feat[223] = float(my_team)
    return feat


# ─── 规则Bot ────────────────────────────────────────
class StrongRuleBot:
    """强规则Bot：小牌优先 + 牌型优先"""
    def select_action(self, obs, return_all_scores=False):
        actions = obs['legal_actions']
        if not actions:
            if return_all_scores:
                return np.zeros(54, dtype=np.int8), []
            return np.zeros(54, dtype=np.int8)
        
        play_actions = [a for a in actions if np.sum(a) > 0]
        if not play_actions:
            if return_all_scores:
                return actions[0], [(actions[0], 0)]
            return actions[0]
        
        # 计算每个动作的分数
        scores = []
        for a in play_actions:
            n = int(np.sum(a))
            ct, cr = recognize_cards(a, obs['trump_rank'])
            
            # 分数 = 牌数*10 + 牌型*5 - 点数(小牌优先)
            rank_score = 0
            for i in range(54):
                if a[i] > 0:
                    rank_score += (i % 13)
            
            score = n * 10 + (ct * 5 if ct else 0) - rank_score * 0.5
            scores.append((a, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        best = scores[0][0]
        
        if return_all_scores:
            return best, scores
        return best


# ─── 训练 ────────────────────────────────────────────
def train():
    device = torch.device('cpu')
    net = PolicyNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG['lr'])
    env = FastDaguaiEnvV2()
    rule_bot = StrongRuleBot()
    
    # 状态
    episode = 0
    win_history = deque(maxlen=100)
    score_history = deque(maxlen=100)  # 得分率：头游+对手垫底
    start_time = time.time()
    save_dir = '/workspace/projects'
    
    # 经验缓冲
    states_buf = []
    actions_buf = []
    scores_buf = []
    
    print(f"🚀 开始训练 v3 (模仿学习 + 强化)")
    print(f"   目标: {CFG['max_episodes']}局")
    
    for ep in range(CFG['max_episodes']):
        obs = env.reset()
        ep_states = []
        ep_actions = []
        ep_scores = []
        
        # 蓝队用规则Bot
        blue_bots = {1: StrongRuleBot(), 3: StrongRuleBot(), 5: StrongRuleBot()}
        
        while not env.done and env.steps < CFG['max_steps']:
            cp = obs['current_player']
            actions = obs['legal_actions']
            
            if not actions:
                break
            
            if cp % 2 == 0:  # 红队：学习
                state = encode_state(
                    obs['hand'], obs['played_all'], obs['current_played'],
                    obs['hand_sizes'], obs['trump_rank'], cp % 2
                )
                
                # 获取规则Bot的最佳动作和所有动作的分数
                best_action, all_scores = rule_bot.select_action(obs, return_all_scores=True)
                
                # 保存经验
                ep_states.append(state)
                ep_actions.append(best_action.copy())
                # 归一化分数作为目标
                if all_scores:
                    max_s = max(s for _, s in all_scores)
                    min_s = min(s for _, s in all_scores)
                    norm_score = (all_scores[0][1] - min_s) / (max_s - min_s + 1e-8)
                else:
                    norm_score = 1.0
                ep_scores.append(norm_score)
                
                action = best_action
            else:  # 蓝队：规则Bot
                action = blue_bots[cp].select_action(obs)
            
            obs, rewards, done, _ = env.step(action)
        
        # 记录结果 - 大怪路子规则：头游获胜
        if env.done and len(env.finish_order) > 0:
            # 头游（第一名）是谁？
            first_player = env.finish_order[0]
            
            # 最后一名：如果只有5人完成，则第6个未完成的玩家是最后一名
            if len(env.finish_order) >= 6:
                last_player = env.finish_order[5]
            else:
                # 找出未完成的那个人
                for i in range(6):
                    if i not in env.finish_order:
                        last_player = i
                        break
                else:
                    last_player = -1
            
            # 头游所在队伍获胜
            if first_player % 2 == 0:  # 红队(0,2,4)
                win_history.append(1.0)
                # 得分：头游是红队，且最后一名是蓝队
                if last_player >= 0 and last_player % 2 == 1:
                    score_history.append(1.0)  # 得分！
                else:
                    score_history.append(0.0)  # 胜但不得分
            else:  # 蓝队(1,3,5)
                win_history.append(0.0)
                # 得分：头游是蓝队，且最后一名是红队
                if last_player >= 0 and last_player % 2 == 0:
                    score_history.append(1.0)  # 对手得分
                else:
                    score_history.append(0.0)
        else:
            win_history.append(0.0)
            score_history.append(0.0)
        
        # 收集经验
        states_buf.extend(ep_states)
        actions_buf.extend(ep_actions)
        scores_buf.extend(ep_scores)
        
        # 批量更新
        if len(states_buf) >= CFG['batch_size']:
            # 随机采样
            idx = np.random.choice(len(states_buf), CFG['batch_size'], replace=False)
            
            states_t = torch.FloatTensor(np.array([states_buf[i] for i in idx])).to(device)
            actions_t = torch.FloatTensor(np.array([actions_buf[i] for i in idx])).to(device)
            targets_t = torch.FloatTensor(np.array([scores_buf[i] for i in idx])).to(device)
            
            # 前向
            preds = net(states_t, actions_t)
            
            # 损失
            loss = F.mse_loss(preds, targets_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 清空部分缓冲
            states_buf = states_buf[-CFG['batch_size']:]
            actions_buf = actions_buf[-CFG['batch_size']:]
            scores_buf = scores_buf[-CFG['batch_size']:]
        
        # 记录状态
        if (ep + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (ep + 1) / elapsed if elapsed > 0 else 0
            win_rate = np.mean(win_history) * 100
            score_rate = np.mean(score_history) * 100  # 得分率
            
            status = {
                'episode': ep + 1,
                'win_rate': round(win_rate, 1),
                'score_rate': round(score_rate, 1),  # 新增得分率
                'avg_reward': round(win_rate / 100, 3),
                'speed': round(speed, 1),
                'elapsed_hours': round(elapsed / 3600, 2),
                'lr': CFG['lr'],
                'temperature': 1.0,
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(os.path.join(save_dir, 'training_status.json'), 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"Ep {ep+1} | 胜率: {win_rate:.1f}% | 速度: {speed:.1f}/s")
        
        # 保存检查点
        if (ep + 1) % CFG['save_interval'] == 0:
            save_path = os.path.join(save_dir, f'model_v3_ep{ep+1}.pt')
            torch.save({
                'model': net.state_dict(),
                'episode': ep + 1,
                'win_rate': np.mean(win_history),
            }, save_path)
            print(f"✅ 保存: {save_path}")
    
    # 最终保存
    torch.save({'model': net.state_dict()}, os.path.join(save_dir, 'model_v3_final.pt'))
    print("🎉 训练完成!")


if __name__ == '__main__':
    train()

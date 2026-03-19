#!/usr/bin/env python3
"""
阶段2训练: 3个AI队友 vs 3随机对手
加载阶段1模型继续训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import json
import time
from collections import deque
from torch.distributions import Categorical

from fast_game_env import FastDaguaiEnv, TEAM_MAP
from card_play_module_v2_optimized import create_fast_card_play_model


# ===================== 配置 =====================
CONFIG = {
    "lr": 1e-4,  # 稍微降低学习率
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "batch_size": 64,
    "update_epochs": 4,
    "total_episodes": 100000,  # 继续训练10万局
    "self_play_games_per_update": 10,
    "save_interval": 500,  # 每500局保存一次
    "eval_interval": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ===================== 经验回放 =====================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        self.advantages = []
        self.returns = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + self.values[t])
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.states)


# ===================== 编码函数 =====================
def encode_hand(hand):
    indices = []
    for c in hand[:27]:
        indices.append(c.get('id', 0) if isinstance(c, dict) else 0)
    while len(indices) < 27:
        indices.append(0)
    return torch.tensor(indices[:27], dtype=torch.long)


def encode_actions(actions):
    candidates = []
    for action in actions[:20]:
        feat = torch.zeros(512)
        if action:
            for i, card in enumerate(action[:5]):
                if isinstance(card, dict):
                    feat[i * 100 + card.get('id', 0) % 100] = 1.0
        candidates.append(feat)
    while len(candidates) < 20:
        candidates.append(torch.zeros(512))
    return torch.stack(candidates)


def encode_game_state(state, player_id):
    features = []
    player_feat = [0] * 6
    player_feat[player_id] = 1
    features.extend(player_feat)
    features.append(len(state.get('finish_order', [])) / 6.0)
    hand_sizes = state.get('hand_sizes', [27] * 6)
    for size in hand_sizes:
        features.append(size / 27.0)
    current_played = state.get('current_played', [])
    type_feat = [0] * 4
    if current_played:
        card_count = len(current_played)
        if card_count in [1, 2, 3]:
            type_feat[card_count - 1] = 1
        elif card_count == 5:
            type_feat[3] = 1
    features.extend(type_feat)
    features.append(state.get('my_team', 0))
    features.append(state.get('trump_rank', 2) / 17.0)
    while len(features) < 64:
        features.append(0)
    return torch.tensor(features[:64], dtype=torch.float32)


# ===================== PPO训练器 =====================
class Stage2Trainer:
    def __init__(self, model, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
        self.buffer = RolloutBuffer()
        self.episode_rewards = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        self.training_stats = {
            'stage': 2,
            'episodes': 0,
            'total_games': 0,
            'wins': 0,
            'avg_reward': 0,
            'win_rate': 0
        }
    
    def select_action(self, hand, legal_actions, game_state, explore=True):
        if not legal_actions:
            return None, 0, 0
        with torch.no_grad():
            hand_tensor = encode_hand(hand).unsqueeze(0).to(self.device)
            action_tensor = encode_actions(legal_actions).unsqueeze(0).to(self.device)
            state_tensor = game_state.unsqueeze(0).to(self.device)
            values = self.model(hand_tensor, action_tensor, state_tensor)
            values = values.squeeze(0)
            valid_values = values[:len(legal_actions)]
            if explore:
                noise = torch.randn_like(valid_values) * 0.1
                valid_values = valid_values + noise
            probs = F.softmax(valid_values, dim=0)
            dist = Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            return action_idx.item(), valid_values[action_idx].item(), log_prob.item()
    
    def collect_episode(self, env, ai_seats=[0, 2, 4]):
        """阶段2: A队3个全是AI，B队随机"""
        state = env.reset()
        step_buffers = {seat: [] for seat in ai_seats}
        
        while not env.done:
            current = state['current_player']
            hand = state['hand']
            legal_actions = state['legal_actions']
            
            if not legal_actions:
                state, rewards, done, _ = env.step([])
                continue
            
            game_state = encode_game_state(state, current)
            
            if current in ai_seats:
                # AI决策
                action_idx, value, log_prob = self.select_action(
                    hand, legal_actions, game_state, explore=True
                )
                action = legal_actions[action_idx]
                
                # 计算每一步的即时奖励
                step_reward = self._calculate_step_reward(
                    action, state, env, current, ai_seats
                )
                
                step_buffers[current].append({
                    'state': game_state,
                    'action': action_idx,
                    'value': value,
                    'log_prob': log_prob,
                    'step_reward': step_reward  # 新增：即时奖励
                })
            else:
                # B队随机
                action = random.choice(legal_actions)
            
            state, rewards, done, _ = env.step(action)
        
        # 计算终局奖励并合并所有AI的经验
        final_rewards = self._calculate_final_rewards(env, ai_seats)
        
        for seat in ai_seats:
            for i, step_data in enumerate(step_buffers[seat]):
                is_last = (i == len(step_buffers[seat]) - 1)
                # 即时奖励 + 终局奖励（最后一步）
                reward = step_data['step_reward'] + (final_rewards[seat] if is_last else 0)
                self.buffer.add(
                    step_data['state'],
                    step_data['action'],
                    reward,
                    step_data['value'],
                    step_data['log_prob'],
                    is_last
                )
        
        return final_rewards[0]  # 返回座位0的奖励用于统计
    
    def _calculate_step_reward(self, action, state, env, seat, ai_seats):
        """
        优化后的每步即时奖励函数
        核心：精准送牌奖励 + 浪费大牌惩罚
        注意：奖励值要小，保持和终局奖励(~0.5-1.0)同量级
        """
        reward = 0.0
        if not action:  # pass
            return reward
        
        # 获取动作的牌面信息
        action_cards = action if isinstance(action, list) else []
        action_ranks = set()
        for c in action_cards:
            if isinstance(c, dict):
                rank = c.get('rank', c.get('value', ''))
                action_ranks.add(str(rank))
        
        # 判断是否是大牌（炸弹、大怪）
        is_big = any(x in ['炸弹', '大怪', '小怪', '2', 'A', 'K'] for x in action_ranks)
        # 判断是否是小牌（单张3-5，对子3-5）
        is_small = any(x in ['3', '4', '5'] for x in action_ranks)
        
        # 获取队友手牌信息
        teammate_seats = [s for s in ai_seats if s != seat]
        teammate_cards = [len(env.hands[s]) for s in teammate_seats]
        teammate_low = any(c < 3 for c in teammate_cards)  # 队友快出完
        
        # ========== 1. 精准送牌奖励（缩小到合理范围） ==========
        if teammate_low:
            # 完美送牌：队友快出完 + 出小牌 + 不出大牌
            if is_small and not is_big:
                reward += 0.3  # 原来是2.0
            # 反面：队友快出完，AI出大牌抢轮次
            elif is_big:
                reward -= 0.2  # 原来是1.5
        
        # ========== 2. 大牌使用优化 ==========
        last_enemy_action = state.get('last_enemy_action', None)
        if last_enemy_action and is_big:
            enemy_ranks = set()
            for c in (last_enemy_action if isinstance(last_enemy_action, list) else []):
                if isinstance(c, dict):
                    r = c.get('rank', c.get('value', ''))
                    enemy_ranks.add(str(r))
            
            # 对手出大牌，AI出大牌压制 → 奖励
            if any(x in ['炸弹', '大怪', '小怪'] for x in enemy_ranks):
                reward += 0.2  # 原来是1.5
            # 对手出小牌，AI出大牌 → 惩罚（浪费！）
            elif any(x in ['3', '4', '5'] for x in enemy_ranks):
                reward -= 0.3  # 原来是2.0
        
        # ========== 3. 不出大牌惩罚 ==========
        if last_enemy_action:
            enemy_ranks = set()
            for c in (last_enemy_action if isinstance(last_enemy_action, list) else []):
                if isinstance(c, dict):
                    r = c.get('rank', c.get('value', ''))
                    enemy_ranks.add(str(r))
            
            if any(x in ['炸弹', '大怪', '小怪'] for x in enemy_ranks):
                my_hand = state.get('hand', [])
                my_ranks = set(str(c.get('rank', c.get('value', ''))) for c in my_hand)
                has_big_in_hand = any(x in ['炸弹', '大怪', '小怪', '2'] for x in my_ranks)
                if has_big_in_hand and not is_big:
                    reward -= 0.25  # 原来是1.8
        
        # ========== 4. 基础组牌奖励 ==========
        if len(action_cards) == 5:
            reward += 0.05
        
        return reward
    
    def _calculate_final_rewards(self, env, ai_seats):
        all_order = list(env.finish_order)
        remaining = [i for i in range(6) if i not in all_order]
        all_order.extend(remaining)
        rank_rewards = [1.0, 0.3, 0.0, -0.3, -0.6, -1.0]
        winner_team = TEAM_MAP[all_order[0]]
        loser_team = TEAM_MAP[all_order[-1]]
        
        rewards = {}
        
        # ========== 【团队配合奖励】核心逻辑 ==========
        # 获取A队3人的排名
        a_team_ranks = [all_order.index(seat) for seat in ai_seats]
        a_team_cards = [len(env.hands[seat]) for seat in ai_seats]
        
        for seat in ai_seats:
            ai_rank = all_order.index(seat)
            reward = rank_rewards[ai_rank]
            
            # 基础团队奖励
            if TEAM_MAP[seat] == winner_team:
                reward += 0.5
            if TEAM_MAP[seat] == loser_team:
                reward -= 0.5
            
            # 1. 【团队包前3奖励】A队3人都进前3名
            if all(r <= 2 for r in a_team_ranks):
                reward += 1.0
            
            # 2. 【队友快走完奖励】队友手牌<3张，自己让路
            teammate_cards = [c for i, c in enumerate(a_team_cards) 
                             if ai_seats[i] != seat]
            if any(c < 3 for c in teammate_cards):
                # 队友快出完了，自己还没出完，让队友先走
                if len(env.hands[seat]) > 3:
                    reward += 0.8  # 让队友奖励
            
            # 3. 【团队整体优势奖励】A队总手牌比B队少
            b_team_cards = sum(len(env.hands[i]) for i in range(6) if i not in ai_seats)
            a_team_total = sum(a_team_cards)
            if a_team_total < b_team_cards:
                reward += 0.3  # 团队领先奖励
            
            # 4. 【助攻奖励】队友头游，自己助攻
            if ai_rank > 0 and a_team_ranks[0] == 0:  # 队友头游
                reward += 0.5
            
            rewards[seat] = reward
        
        return rewards
    
    def update(self):
        if len(self.buffer) == 0:
            return 0, 0, 0
        
        self.buffer.compute_gae(CONFIG["gamma"], CONFIG["gae_lambda"])
        
        states = torch.stack(self.buffer.states).to(self.device)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(self.buffer.returns, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_value_loss = 0
        
        for _ in range(CONFIG["update_epochs"]):
            batch_size = states.size(0)
            dummy_hand = torch.zeros(batch_size, 27, dtype=torch.long, device=self.device)
            dummy_actions = torch.randn(batch_size, 20, 512, device=self.device)
            
            new_values = self.model(dummy_hand, dummy_actions, states)
            new_values = new_values.mean(dim=1)
            
            value_loss = F.mse_loss(new_values, returns)
            entropy = -torch.mean(torch.log(torch.abs(new_values) + 1e-8))
            entropy_loss = -CONFIG["entropy_coef"] * entropy
            
            loss = value_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_value_loss += value_loss.item()
        
        self.buffer.clear()
        return total_loss / CONFIG["update_epochs"], 0, total_value_loss / CONFIG["update_epochs"]
    
    def evaluate(self, num_games=50):
        wins = 0
        ai_seats = [0, 2, 4]
        
        for _ in range(num_games):
            env = FastDaguaiEnv(max_actions_per_step=30)
            state = env.reset()
            
            while not env.done:
                current = state['current_player']
                legal_actions = state['legal_actions']
                
                if not legal_actions:
                    state, _, done, _ = env.step([])
                    continue
                
                if current in ai_seats:
                    game_state = encode_game_state(state, current)
                    action_idx, _, _ = self.select_action(
                        state['hand'], legal_actions, game_state, explore=False
                    )
                    action = legal_actions[action_idx] if action_idx < len(legal_actions) else legal_actions[0]
                else:
                    action = random.choice(legal_actions)
                
                state, _, done, _ = env.step(action)
            
            all_order = list(env.finish_order)
            remaining = [i for i in range(6) if i not in all_order]
            all_order.extend(remaining)
            winner_team = TEAM_MAP[all_order[0]]
            if TEAM_MAP[0] == winner_team:
                wins += 1
        
        return wins / num_games


# ===================== 主训练函数 =====================
def train_stage2():
    log_dir = '/Users/sandro/Documents/大怪路子/ai_training/self_play_logs'
    checkpoint_dir = '/Users/sandro/Documents/大怪路子AI版/train'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 加载stage2_final.pt继续训练
    model = create_fast_card_play_model(CONFIG['device'])
    stage2_model = '/Users/sandro/Documents/大怪路子AI版/train/stage2_final.pt'
    
    if os.path.exists(stage2_model):
        checkpoint = torch.load(stage2_model, map_location=CONFIG['device'], weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"从stage2模型继续训练: {stage2_model}", flush=True)
        if 'win_rate' in checkpoint:
            print(f"之前胜率: {checkpoint['win_rate']:.1%}", flush=True)
        if 'episode' in checkpoint:
            print(f"之前训练局数: {checkpoint['episode']}", flush=True)
    
    trainer = Stage2Trainer(model, CONFIG['device'])
    stats = trainer.training_stats
    
    print("\n" + "="*60, flush=True)
    print("阶段2: 3个AI队友 vs 3随机对手", flush=True)
    print("A队: 座位0,2,4 全是AI", flush=True)
    print("B队: 座位1,3,5 随机出牌", flush=True)
    print("="*60, flush=True)
    
    start_time = time.time()
    
    for episode in range(1, CONFIG['total_episodes'] + 1):
        env = FastDaguaiEnv(max_actions_per_step=30)
        reward = trainer.collect_episode(env, ai_seats=[0, 2, 4])
        
        stats['episodes'] = episode
        stats['total_games'] += 1
        trainer.episode_rewards.append(reward)
        
        # 判断胜负（以座位0为准）
        all_order = list(env.finish_order)
        remaining = [i for i in range(6) if i not in all_order]
        all_order.extend(remaining)
        winner_team = TEAM_MAP[all_order[0]]
        if TEAM_MAP[0] == winner_team:
            stats['wins'] += 1
        
        if episode % CONFIG['self_play_games_per_update'] == 0:
            trainer.update()
        
        if episode % CONFIG['save_interval'] == 0:
            win_rate = stats['wins'] / stats['total_games']
            avg_reward = np.mean(trainer.episode_rewards)
            
            save_path = f'{checkpoint_dir}/stage2_ep{episode}.pt'
            torch.save({
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'win_rate': win_rate,
                'avg_reward': avg_reward
            }, save_path)
            
            stats['win_rate'] = win_rate
            stats['avg_reward'] = avg_reward
            
            with open(f'{log_dir}/stage2_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            elapsed = time.time() - start_time
            print(f"[阶段2] Episode {episode}/{CONFIG['total_episodes']} | "
                  f"WinRate: {win_rate:.1%} | AvgReward: {avg_reward:.3f} | "
                  f"Time: {elapsed:.0f}s", flush=True)
        
        if episode % CONFIG['eval_interval'] == 0:
            eval_win_rate = trainer.evaluate(num_games=50)
            print(f"  [评估] vs随机对手胜率: {eval_win_rate:.1%}", flush=True)
    
    # 保存最终模型
    final_path = f'{checkpoint_dir}/stage2_final.pt'
    torch.save({
        'stage': 2,
        'model_state_dict': model.state_dict(),
        'win_rate': stats['win_rate'],
        'avg_reward': stats['avg_reward']
    }, final_path)
    
    # 最终评估
    eval_win_rate = trainer.evaluate(num_games=100)
    
    print(f"\n" + "="*60, flush=True)
    print("阶段2训练完成!", flush=True)
    print(f"最终胜率: {stats['win_rate']:.1%}", flush=True)
    print(f"vs随机对手胜率: {eval_win_rate:.1%}", flush=True)
    print(f"模型已保存: {final_path}", flush=True)
    print("="*60, flush=True)


if __name__ == '__main__':
    train_stage2()

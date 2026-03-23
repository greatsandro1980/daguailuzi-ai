#!/usr/bin/env python3
"""
V15 - 综合优化版
1. Transformer网络架构
2. 课程学习 (Curriculum Learning)
3. 多样化对手
4. 优先经验回放
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import functools
import random
from collections import deque
print = functools.partial(print, flush=True)

# ============== 游戏环境 ==============
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.trick_count = 0
        
    def reset(self):
        deck = [i for i in range(13) for _ in range(4)] + [13, 14]
        np.random.shuffle(deck)
        self.hands = np.zeros((6, 15), dtype=np.int32)
        idx = 0
        for p in range(6):
            count = 9 if p < 4 else 8
            for _ in range(count):
                self.hands[p, deck[idx]] += 1
                idx += 1
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.trick_count = 0
        return self.get_state()
    
    def get_state(self):
        s = np.zeros(60, dtype=np.float32)
        s[:15] = self.hands[self.current]
        if self.last_play is not None:
            s[15:30] = self.last_play
        for i in range(6):
            s[30+i] = self.hands[i].sum() / 27.0
        if self.last_play is not None and self.last_play.sum() > 0:
            cnt = int(self.last_play[self.last_play > 0][0])
            s[36 + min(cnt-1, 2)] = 1.0
        s[39] = 1.0 if self.last_play is None else 0.0
        team = self.current % 2
        for i, p in enumerate([p for p in range(6) if p % 2 == team and p != self.current]):
            s[40+i] = 1.0 if self.finished[p] else 0.0
        for i, p in enumerate([p for p in range(6) if p % 2 != team]):
            s[43+i] = 1.0 if self.finished[p] else 0.0
        hand = self.hands[self.current]
        s[46] = sum(1 for i in range(13) if hand[i] == 1)
        s[47] = sum(1 for i in range(13) if hand[i] == 2)
        s[48] = sum(1 for i in range(13) if hand[i] >= 3)
        s[49] = hand[13]
        s[50] = hand[14]
        s[51] = hand.sum() / 27.0
        s[52] = self.trick_count / 15.0
        return s
    
    def get_actions(self):
        hand = self.hands[self.current]
        actions = []
        if self.last_play is None or self.last_player == self.current:
            for i in range(15):
                if hand[i] >= 1: actions.append((i, 1))
            for i in range(13):
                if hand[i] >= 2: actions.append((i, 2))
            for i in range(13):
                if hand[i] >= 3: actions.append((i, 3))
        else:
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt: actions.append((i, last_cnt))
            actions.append((-1, 0))
        return actions if actions else [(-1, 0)]
    
    def step(self, card, cnt):
        reward = 0.0
        hand = self.hands[self.current]
        team = self.current % 2
        
        if card >= 0:
            hand[card] -= cnt
            reward += 0.1 + cnt * 0.1
            
            # 策略奖励
            if self.last_play is None or self.last_player == self.current:
                if card <= 5:
                    reward += 0.05
                elif card >= 13:
                    reward -= 0.1
            
            if card == 14 and self.trick_count < 3:
                reward -= 0.05
            elif card == 13 and self.trick_count < 2:
                reward -= 0.03
            
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[card] = cnt
            self.last_player = self.current
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None else 0
                if last_val >= 10:
                    reward += 0.05
            self.passes += 1
        
        for _ in range(6):
            self.current = (self.current + 1) % 6
            if not self.finished[self.current]:
                break
        
        if self.passes >= sum(1 for p in range(6) if not self.finished[p]):
            self.last_play = None
            self.last_player = -1
            self.passes = 0
            self.trick_count += 1
        
        team0_done = all(self.finished[p] for p in [0, 2, 4])
        team1_done = all(self.finished[p] for p in [1, 3, 5])
        
        if team0_done or team1_done:
            winner = 0 if team0_done else 1
            if winner == 0:
                reward += 2.0
            return True, winner, reward
        
        return False, -1, reward


# ============== Transformer网络 ==============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TransformerActorCritic(nn.Module):
    """基于Transformer的策略网络"""
    def __init__(self, state_dim=60, action_dim=16, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        
        # 输入嵌入
        self.input_embed = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 策略头
        self.actor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 价值头
        self.critic = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch, state_dim) 或 (state_dim,)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 嵌入并添加位置编码
        x = self.input_embed(x)  # (batch, d_model)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        
        # 输出
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value
    
    def get_action(self, state, actions, deterministic=True):
        with torch.no_grad():
            logits, value = self(torch.FloatTensor(state))
        
        # logits形状: (1, action_dim) 或 (action_dim,)
        if logits.dim() == 2:
            logits = logits.squeeze(0)
        
        valid = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits, 0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        valid_probs = valid_probs / valid_probs.sum()
        
        if deterministic:
            a_idx = valid_probs.argmax().item()
        else:
            a_idx = torch.multinomial(valid_probs, 1).item()
        
        return a_idx, valid[a_idx], value.item() if isinstance(value, torch.Tensor) else value


# ============== 多样化对手 ==============
class OpponentPool:
    """多样化对手池"""
    def __init__(self):
        self.strategies = {
            'random': self._random_action,
            'rule': self._rule_action,
            'aggressive': self._aggressive_action,
            'defensive': self._defensive_action,
            'smart_random': self._smart_random,
        }
    
    def get_action(self, game, strategy='rule'):
        return self.strategies.get(strategy, self._rule_action)(game)
    
    def _random_action(self, game):
        actions = game.get_actions()
        return np.random.randint(len(actions))
    
    def _rule_action(self, game):
        actions = game.get_actions()
        valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
        if not valid: return len(actions) - 1
        # 优先出对子/三张
        multi = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
        if multi:
            multi.sort(key=lambda x: x[1])
            return multi[0][0]
        valid.sort(key=lambda x: x[1])
        return valid[0][0]
    
    def _aggressive_action(self, game):
        """激进策略：优先出大牌"""
        actions = game.get_actions()
        valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
        if not valid: return len(actions) - 1
        # 优先出大牌
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    
    def _defensive_action(self, game):
        """防守策略：保留大牌"""
        actions = game.get_actions()
        valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
        if not valid: return len(actions) - 1
        # 避免出王
        non_joker = [(i, c, cnt) for i, c, cnt in valid if c < 13]
        if non_joker:
            non_joker.sort(key=lambda x: x[1])
            return non_joker[0][0]
        valid.sort(key=lambda x: x[1])
        return valid[0][0]
    
    def _smart_random(self, game):
        """智能随机：70%规则 + 30%随机"""
        if np.random.random() < 0.7:
            return self._rule_action(game)
        return self._random_action(game)


# ============== 优先经验回放 ==============
class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
    
    def push(self, state, action, reward, value, log_prob):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        
        self.buffer.append({
            'state': state, 'action': action, 'reward': reward,
            'value': value, 'log_prob': log_prob
        })
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = [self.buffer[i] for i in indices]
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


# ============== 测试函数 ==============
def test(model, n=500, opp='rule'):
    game = Game()
    opponent = OpponentPool()
    wins = 0
    
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx, _, _ = model.get_action(game.get_state(), actions, deterministic=True)
            else:
                idx = opponent.get_action(game, opp)
            done, winner, _ = game.step(*actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== 课程学习训练 ==============
def train_curriculum(model, n_games=10000):
    game = Game()
    opponent = OpponentPool()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000)
    
    # 经验回放
    replay_buffer = PrioritizedReplayBuffer()
    
    best_rule = 0
    best_random = 0
    best_combined = 0
    
    # 课程学习阶段定义
    # 阶段0: 70%随机, 30%规则 (简单)
    # 阶段1: 50%随机, 50%规则 (中等)
    # 阶段2: 30%随机, 70%规则 (困难)
    # 阶段3: 混合对手 (最难)
    def get_opponent(stage):
        if stage == 0:
            return np.random.choice(['random', 'random', 'random', 'rule', 'rule'])
        elif stage == 1:
            return np.random.choice(['random', 'random', 'rule', 'rule', 'rule'])
        elif stage == 2:
            return np.random.choice(['random', 'rule', 'rule', 'rule', 'smart_random'])
        else:
            return np.random.choice(['rule', 'smart_random', 'aggressive', 'defensive'])
    
    print(f"\n课程学习训练 ({n_games}局)...")
    print("=" * 60)
    print("阶段0: 随机70% + 规则30%")
    print("阶段1: 随机50% + 规则50%")
    print("阶段2: 随机30% + 规则70%")
    print("阶段3: 多样化混合对手")
    print("=" * 60)
    
    episode_memory = []
    
    for g in range(n_games):
        # 确定当前阶段
        progress = g / n_games
        if progress < 0.25:
            stage = 0
        elif progress < 0.5:
            stage = 1
        elif progress < 0.75:
            stage = 2
        else:
            stage = 3
        
        game.reset()
        episode_data = []
        
        while True:
            state = game.get_state()
            actions = game.get_actions()
            
            if game.current % 2 == 0:
                # 我方使用模型
                idx, action, value = model.get_action(state, actions, deterministic=False)
                with torch.no_grad():
                    logits, _ = model(torch.FloatTensor(state))
                    if logits.dim() == 2:
                        logits = logits.squeeze(0)
                log_prob = torch.log_softmax(logits, 0)[action].item()
                episode_data.append({
                    'state': state, 'action': action, 'value': value, 'log_prob': log_prob
                })
                card, cnt = actions[idx]
            else:
                # 对方使用课程策略
                opp = get_opponent(stage)
                idx = opponent.get_action(game, opp)
                card, cnt = actions[idx]
            
            done, winner, reward = game.step(card, cnt)
            if episode_data:
                episode_data[-1]['reward'] = reward
            
            if done:
                # 计算returns
                returns = []
                R = 0
                for data in reversed(episode_data):
                    R = data.get('reward', 0) + 0.99 * R
                    returns.insert(0, R)
                
                for i, data in enumerate(episode_data):
                    replay_buffer.push(
                        data['state'], data['action'], returns[i],
                        data['value'], data['log_prob']
                    )
                break
        
        # 定期更新
        if len(replay_buffer.buffer) >= 2048 and (g + 1) % 4 == 0:
            batch, indices, weights = replay_buffer.sample(1024)
            
            states = torch.FloatTensor(np.array([d['state'] for d in batch]))
            actions = torch.LongTensor([d['action'] for d in batch])
            old_log_probs = torch.FloatTensor([d['log_prob'] for d in batch])
            returns = torch.FloatTensor([d['reward'] for d in batch])
            old_values = torch.FloatTensor([d['value'] for d in batch])
            weights = torch.FloatTensor(weights)
            
            # 优势函数
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO更新
            for _ in range(3):
                logits, new_values = model(states)
                new_log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.75, 1.25) * advantages
                policy_loss = -(torch.min(surr1, surr2) * weights).mean()
                value_loss = (nn.MSELoss(reduction='none')(new_values.squeeze(), returns) * weights).mean()
                entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.02 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            # 更新优先级
            with torch.no_grad():
                new_priorities = torch.abs(advantages).numpy() + 0.01
            replay_buffer.update_priorities(indices, new_priorities)
            
            scheduler.step()
        
        # 定期测试
        if (g + 1) % 500 == 0:
            stage_name = ['随机70%', '随机50%', '随机30%', '混合对手'][stage]
            print(f"\n  [{g+1:>5,}局 | 阶段{stage}: {stage_name}]")
            
            r1 = test(model, 500, 'rule')
            r2 = test(model, 500, 'random')
            r3 = test(model, 300, 'smart_random')
            combined = r1 + r2
            
            print(f"    vs规则: {r1:.1f}% | vs随机: {r2:.1f}% | vs智能: {r3:.1f}% | 综合: {combined:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v15_best.pt')
                print(f"    ★ 新最佳!")
            
            if r1 >= 90 and r2 >= 90:
                print(f"    🎯 目标达成!")
                break
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V15 - 综合优化版")
    print("目标：vs规则 ≥90%, vs随机 ≥90%")
    print("=" * 60)
    print("\n优化技术:")
    print("  1. Transformer网络架构")
    print("  2. 课程学习 (Curriculum Learning)")
    print("  3. 多样化对手池")
    print("  4. 优先经验回放")
    print("=" * 60)
    
    # 创建模型
    model = TransformerActorCritic(
        state_dim=60,
        action_dim=16,
        d_model=256,
        nhead=4,
        num_layers=3
    )
    
    # 尝试加载预训练权重（如果兼容）
    if os.path.exists('/workspace/projects/rl_v14b_best.pt'):
        print("\n尝试加载V14b预训练嵌入...")
        try:
            old_state = torch.load('/workspace/projects/rl_v14b_best.pt', weights_only=True)
            # 只加载输入嵌入层
            model.input_embed.weight.data[:, :60] = old_state['shared.0.weight']
            print("预训练嵌入加载成功!")
        except:
            print("预训练权重不兼容，从头训练")
    
    # 初始测试
    print("\n初始测试...")
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 训练
    model, best_rule, best_random = train_curriculum(model, n_games=10000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试 (1000局):")
    
    if os.path.exists('/workspace/projects/rl_v15_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v15_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    r3 = test(model, 500, 'smart_random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    print(f"  vs智能Bot: {r3:.1f}%")
    print(f"  综合得分: {r1+r2:.1f}%")
    
    if r1 >= 90 and r2 >= 90:
        print("\n🎯 目标达成!")
    else:
        print(f"\n最佳结果: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%")


if __name__ == '__main__':
    main()

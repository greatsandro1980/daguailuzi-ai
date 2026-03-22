#!/usr/bin/env python3
"""
强化学习训练 V5 - DQN版本
改进：
1. DQN算法（经验回放 + 目标网络）
2. Dueling Network架构
3. 更丰富的状态表示（200维）
4. 阶段性奖励设计
5. 渐进式对手训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
import random
from datetime import datetime
from collections import deque

# ============== 游戏环境 ==============
class FastGame:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.last_type = 0  # 0:首发, 1:单张, 2:对子, 3:三张
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        self.played_cards = np.zeros((6, 15), dtype=np.int32)  # 历史出牌
        self.round = 0
        
    def reset(self):
        # 52张普通牌 + 2张王
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
        self.last_type = 0
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        self.played_cards = np.zeros((6, 15), dtype=np.int32)
        self.round = 0
    
    def get_state(self, player=None):
        """获取丰富的状态表示 (200维)"""
        if player is None:
            player = self.current
            
        s = np.zeros(200, dtype=np.float32)
        offset = 0
        
        # 自己手牌 (15)
        s[offset:offset+15] = self.hands[player] / 4.0
        offset += 15
        
        # 上一轮出牌 (15)
        if self.last_play is not None:
            s[offset:offset+15] = self.last_play / 4.0
        offset += 15
        
        # 所有玩家剩余牌数 (6)
        for i in range(6):
            s[offset+i] = self.hands[i].sum() / 10.0
        offset += 6
        
        # 队友剩余牌 (3)
        team = player % 2
        for i, mate in enumerate([p for p in range(6) if p % 2 == team and p != player]):
            s[offset+i] = self.hands[mate].sum() / 10.0
        offset += 3
        
        # 对手剩余牌 (3)
        for i, opp in enumerate([p for p in range(6) if p % 2 != team]):
            s[offset+i] = self.hands[opp].sum() / 10.0
        offset += 3
        
        # 当前出牌类型 (3): one-hot
        if self.last_type > 0:
            s[offset + self.last_type - 1] = 1.0
        offset += 3
        
        # 是否轮到自己 (1)
        s[offset] = 1.0 if player == self.current else 0.0
        offset += 1
        
        # 是否首发 (1)
        s[offset] = 1.0 if self.last_play is None else 0.0
        offset += 1
        
        # 自己已出过的牌 (15)
        s[offset:offset+15] = self.played_cards[player] / 4.0
        offset += 15
        
        # 队友已出过的牌 (45 = 3×15)
        for i, mate in enumerate([p for p in range(6) if p % 2 == team and p != player]):
            s[offset+i*15:offset+(i+1)*15] = self.played_cards[mate] / 4.0
        offset += 45
        
        # 对手已出过的牌 (45)
        for i, opp in enumerate([p for p in range(6) if p % 2 != team]):
            s[offset+i*15:offset+(i+1)*15] = self.played_cards[opp] / 4.0
        offset += 45
        
        # 已出完的玩家 (6)
        for i in range(6):
            s[offset+i] = 1.0 if self.finished[i] else 0.0
        offset += 6
        
        # 大小王状态 (2)
        s[offset] = 1.0 if self.hands[player, 13] > 0 else 0.0
        s[offset+1] = 1.0 if self.hands[player, 14] > 0 else 0.0
        offset += 2
        
        # 剩余牌型统计 (5): 单张数、对子数、三张数、单王、双王
        hand = self.hands[player]
        s[offset] = sum(1 for i in range(13) if hand[i] == 1) / 13.0
        s[offset+1] = sum(1 for i in range(13) if hand[i] == 2) / 6.0
        s[offset+2] = sum(1 for i in range(13) if hand[i] >= 3) / 4.0
        s[offset+3] = hand[13] / 2.0
        s[offset+4] = hand[14] / 2.0
        offset += 5
        
        # 回合数归一化 (1)
        s[offset] = min(self.round / 100.0, 1.0)
        
        return s
    
    def get_actions(self):
        """获取合法动作列表，返回动作索引"""
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
            # 首发：可以选择任意牌型
            for i in range(15):
                if hand[i] >= 1:
                    actions.append((i, 1))  # 单张
            for i in range(13):
                if hand[i] >= 2:
                    actions.append((i, 2))  # 对子
            for i in range(13):
                if hand[i] >= 3:
                    actions.append((i, 3))  # 三张
        else:
            # 跟牌：必须出更大的同类型牌
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt:
                    actions.append((i, last_cnt))
            actions.append((-1, 0))  # 过牌
        
        return actions if actions else [(-1, 0)]
    
    def encode_action(self, card, count):
        """将动作编码为Q值索引 (0-44)
        0-14: 单张 (0-12普通牌, 13小王, 14大王)
        15-27: 对子 (0-12普通牌)
        28-40: 三张 (0-12普通牌)
        41-44: 过牌等特殊动作
        """
        if card < 0:  # 过牌
            return 41
        if count == 1:  # 单张
            return card
        elif count == 2:  # 对子
            return 15 + card
        elif count == 3:  # 三张
            return 28 + card
        return 41  # 默认过牌
    
    def decode_action(self, action_id, actions):
        """将动作索引解码为(牌, 数量)"""
        for card, count in actions:
            if self.encode_action(card, count) == action_id:
                return card, count
        # 如果不匹配，返回第一个合法动作
        return actions[0] if actions else (-1, 0)
    
    def step(self, card, count):
        """执行动作，返回(done, winner, reward)"""
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
            self.played_cards[self.current, card] += count
            self.round += 1
        
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        
        # 计算即时奖励
        reward = 0.0
        if count > 0:
            reward += 0.02 * count  # 出牌奖励
            if count >= 2:
                reward += 0.03  # 对子/三张奖励
        
        # 检查出完
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            self.finish_order.append(self.current)
            
            # 头游奖励
            if len(self.finish_order) == 1:
                reward += 0.5 if self.current % 2 == 0 else -0.3
            
            # 队友出完奖励
            if self.current % 2 == 0:
                reward += 0.1
            
            # 全部出完
            if all(self.finished):
                winner = self.finish_order[0] % 2
                final_reward = 1.0 if winner == 0 else -0.5
                return True, winner, reward + final_reward
        
        if action.sum() > 0:
            self.last_play = action.copy()
            self.last_player = self.current
            self.last_type = count if count > 0 else 1
            self.passes = 0
        else:
            self.passes += 1
        
        if self.passes >= 5:
            self.last_play = None
            self.last_player = -1
            self.last_type = 0
            self.passes = 0
        
        self.current = (self.current + 1) % 6
        while self.finished[self.current]:
            self.current = (self.current + 1) % 6
        
        return False, -1, reward


# ============== Dueling DQN 网络 ==============
class DuelingDQN(nn.Module):
    def __init__(self, state_dim=200, action_dim=42, hidden_dim=256):
        super().__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态价值流 (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 动作优势流 (A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ============== Replay Buffer ==============
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============== DQN Agent ==============
class DQNAgent:
    def __init__(self, state_dim=200, action_dim=42, hidden_dim=256, lr=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.action_dim = action_dim
        self.update_count = 0
        self.target_update_freq = 500
    
    def select_action(self, state, actions, eval_mode=False):
        """选择动作"""
        # 编码动作为Q值索引
        def encode_action(card, count):
            if card < 0:
                return 41
            if count == 1:
                return card
            elif count == 2:
                return 15 + card
            elif count == 3:
                return 28 + card
            return 41
        
        valid_action_ids = [encode_action(c, cnt) for c, cnt in actions]
        
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.choice(valid_action_ids)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)
        
        valid_q = [(aid, q_values[aid].item()) for aid in valid_action_ids]
        best_action = max(valid_q, key=lambda x: x[1])[0]
        return best_action
    
    def update(self, batch_size=64):
        """训练一步"""
        if len(self.memory) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 用policy_net选动作，用target_net评估
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络 - 更频繁更新提高稳定性
        self.update_count += 1
        if self.update_count % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """每个episode结束后衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.1)


# ============== 对手策略 ==============
def random_action(actions):
    return np.random.randint(len(actions))

def rule_action(actions, game):
    """智能规则Bot"""
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    
    if not valid:
        return len(actions) - 1  # 过牌
    
    hand = game.hands[game.current]
    
    # 策略：根据剩余牌数调整
    remaining = hand.sum()
    
    if remaining <= 3:
        # 剩余牌少，优先出大牌快速出完
        valid.sort(key=lambda x: -x[1])  # 大牌优先
    else:
        # 剩余牌多，优先出对子和三张
        pairs = [(i, card, count) for i, card, count in valid if count >= 2]
        if pairs:
            pairs.sort(key=lambda x: x[1])  # 小牌优先
            return pairs[0][0]
        valid.sort(key=lambda x: x[1])  # 单张小牌优先
    
    return valid[0][0]

def strong_rule_action(actions, game):
    """更强的规则Bot"""
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    
    if not valid:
        return len(actions) - 1
    
    hand = game.hands[game.current]
    team = game.current % 2
    
    # 检查队友状态
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    opps_done = sum(1 for p in range(6) if p % 2 != team and game.finished[p])
    
    # 如果队友都出完了，激进出牌
    if mates_done >= 2:
        valid.sort(key=lambda x: -x[1])  # 大牌优先
        return valid[0][0]
    
    # 如果对手出完多，保守出牌
    if opps_done >= 2:
        # 优先对子三张
        pairs = [(i, card, count) for i, card, count in valid if count >= 2]
        if pairs:
            pairs.sort(key=lambda x: x[1])
            return pairs[0][0]
    
    # 默认：对子/三张优先，小牌优先
    pairs = [(i, card, count) for i, card, count in valid if count >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


# ============== 训练函数 ==============
def play_episode(agent, game, opponent_type='mixed', eval_mode=False):
    """进行一局游戏"""
    game.reset()
    total_reward = 0
    transitions = []
    
    while True:
        player = game.current
        actions = game.get_actions()
        state = game.get_state()
        
        if player % 2 == 0:  # 红队：模型
            action_id = agent.select_action(state, actions, eval_mode)
            card, count = game.decode_action(action_id, actions)
        else:  # 蓝队：对手
            if opponent_type == 'random':
                idx = random_action(actions)
            elif opponent_type == 'strong':
                idx = strong_rule_action(actions, game)
            else:  # mixed or rule
                if opponent_type == 'mixed' and np.random.random() < 0.3:
                    idx = random_action(actions)
                else:
                    idx = rule_action(actions, game)
            card, count = actions[idx]
            action_id = game.encode_action(card, count)
        
        done, winner, reward = game.step(card, count)
        next_state = game.get_state() if not done else np.zeros(200)
        
        if player % 2 == 0:
            total_reward += reward
            transitions.append((state, action_id, reward, next_state, done))
            
            if not eval_mode:
                agent.memory.push(state, action_id, reward, next_state, done)
        
        if done:
            return winner, total_reward, transitions, game.finish_order


def test_agent(agent, game, num_episodes=100, opponent='rule'):
    """测试胜率"""
    wins = 0
    scores = 0
    
    for _ in range(num_episodes):
        winner, _, _, finish_order = play_episode(agent, game, opponent, eval_mode=True)
        if winner == 0:
            wins += 1
            # 得分：头游是红队且最后一名是蓝队
            if finish_order and finish_order[-1] % 2 == 1:
                scores += 1
    
    return wins / num_episodes * 100, scores / num_episodes * 100


def main():
    print("=" * 70)
    print("强化学习训练 V5 - DQN版本")
    print("改进：Dueling DQN + 经验回放 + 阶段性奖励")
    print("=" * 70)
    
    # 初始化 - 更保守的参数确保稳定训练
    agent = DQNAgent(
        state_dim=200,
        action_dim=42,
        hidden_dim=256,
        lr=5e-5,           # 更低的学习率
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.15,  # 更高的最小探索率
        epsilon_decay=0.99995  # 更慢的衰减
    )
    game = FastGame()
    
    # 尝试加载已有模型
    model_path = '/workspace/projects/rl_v5.pt'
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ 加载模型: {model_path}")
    
    # 预热：用随机策略填充经验回放
    print("预热中...")
    for _ in range(2000):
        game.reset()
        while True:
            player = game.current
            actions = game.get_actions()
            state = game.get_state()
            
            # 随机选择动作
            idx = np.random.randint(len(actions))
            card, count = actions[idx]
            action_id = game.encode_action(card, count)
            
            done, winner, reward = game.step(card, count)
            next_state = game.get_state() if not done else np.zeros(200)
            
            if player % 2 == 0:  # 只存储红队经验
                agent.memory.push(state, action_id, reward, next_state, done)
            
            if done:
                break
    print(f"预热完成，经验池大小: {len(agent.memory)}")
    
    target_episodes = 100000
    test_interval = 1000
    save_interval = 5000
    
    # 统计
    total_wins = 0
    total_games = 0
    total_scores = 0
    recent_wins = deque(maxlen=100)
    recent_scores = deque(maxlen=100)
    best_win_rate = 0
    start_time = time.time()
    
    print(f"\n开始训练: 0 -> {target_episodes:,} 局")
    print("-" * 70)
    print(f"{'局数':>10} | {'vs规则':>8} | {'vs随机':>8} | {'vs强敌':>8} | {'最佳':>8} | {'ε':>6} | {'速度':>8}")
    print("-" * 70)
    
    for ep in range(target_episodes):
        # 渐进式对手：初期随机多，后期规则多
        progress = ep / target_episodes
        if progress < 0.3:
            opponent = 'mixed'  # 30%随机
        elif progress < 0.6:
            opponent = 'rule'   # 纯规则
        else:
            opponent = 'strong' # 强规则
        
        winner, reward, transitions, finish_order = play_episode(agent, game, opponent)
        
        # 训练 - 每局只训练1-2步，避免过拟合
        if len(agent.memory) >= 1000:
            for _ in range(2):
                agent.update(batch_size=64)
        
        # 每局结束后衰减探索率
        agent.decay_epsilon()
        
        # 统计
        total_games += 1
        if winner == 0:
            total_wins += 1
            recent_wins.append(1)
            if finish_order and finish_order[-1] % 2 == 1:
                total_scores += 1
                recent_scores.append(1)
            else:
                recent_scores.append(0)
        else:
            recent_wins.append(0)
            recent_scores.append(0)
        
        # 测试
        if (ep + 1) % test_interval == 0:
            rate_rule, score_rule = test_agent(agent, game, 100, 'rule')
            rate_random, score_random = test_agent(agent, game, 100, 'random')
            rate_strong, _ = test_agent(agent, game, 100, 'strong')
            
            elapsed = time.time() - start_time
            speed = (ep + 1) / elapsed
            
            best = max(rate_rule, best_win_rate)
            if rate_rule > best_win_rate:
                best_win_rate = rate_rule
                agent.save('/workspace/projects/rl_v5_best.pt')
            
            total_win_rate = total_wins / total_games * 100
            avg_score_rate = sum(recent_scores) / len(recent_scores) * 100 if recent_scores else 0
            
            print(f"{ep+1:>10,} | {rate_rule:>7.1f}% | {rate_random:>7.1f}% | {rate_strong:>7.1f}% | {best_win_rate:>7.1f}% | {agent.epsilon:.3f} | {speed:>6.0f}局/秒")
            
            # 保存状态
            status = {
                'episode': ep + 1,
                'target': target_episodes,
                'progress': (ep + 1) / target_episodes * 100,
                'win_rate': round(sum(recent_wins)/len(recent_wins)*100, 1) if recent_wins else 0,
                'total_win_rate': round(total_win_rate, 1),
                'score_rate': round(avg_score_rate, 1),
                'vs_rule': round(rate_rule, 1),
                'vs_random': round(rate_random, 1),
                'vs_strong': round(rate_strong, 1),
                'best_rate': round(best_win_rate, 1),
                'epsilon': round(agent.epsilon, 4),
                'total_wins': total_wins,
                'total_games': total_games,
                'speed': round(speed, 1),
                'elapsed_hours': round(elapsed / 3600, 2),
                'training_type': 'reinforcement_learning_v5_dqn',
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
        
        # 保存模型
        if (ep + 1) % save_interval == 0:
            agent.save(model_path)
    
    # 训练完成
    print("-" * 70)
    print(f"训练完成！最佳胜率: {best_win_rate:.1f}%")
    agent.save('/workspace/projects/rl_v5_final.pt')
    
    # 最终测试
    final_rule, _ = test_agent(agent, game, 200, 'rule')
    final_random, _ = test_agent(agent, game, 200, 'random')
    final_strong, _ = test_agent(agent, game, 200, 'strong')
    print(f"最终测试: vs规则Bot {final_rule:.1f}%, vs随机Bot {final_random:.1f}%, vs强敌 {final_strong:.1f}%")


if __name__ == '__main__':
    main()

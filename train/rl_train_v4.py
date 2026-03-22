#!/usr/bin/env python3
"""
强化学习训练 V4 - 改进版
改进：
1. 更好的奖励函数
2. 更强的网络结构
3. 更丰富的状态表示
4. 混合对手训练（随机+规则Bot）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import deque

# ============== 改进的游戏环境 ==============
class FastGame:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        self.cards_played = [0, 0]  # 双方出牌数
        
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
        self.finish_order = []
        self.cards_played = [0, 0]
    
    def get_state(self, player=None):
        """获取更丰富的状态表示"""
        if player is None:
            player = self.current
            
        s = np.zeros(120, dtype=np.float32)
        
        # 当前玩家手牌 (15)
        s[:15] = self.hands[player] / 4.0
        
        # 上一轮出牌 (15)
        if self.last_play is not None:
            s[15:30] = self.last_play / 4.0
        
        # 所有玩家剩余牌数 (6)
        for i in range(6):
            s[30+i] = self.hands[i].sum() / 10.0
        
        # 队友信息 (3个队友的剩余牌)
        team = player % 2
        for i, mate in enumerate([p for p in range(6) if p % 2 == team and p != player]):
            s[36+i] = self.hands[mate].sum() / 10.0
        
        # 对手信息 (3个对手的剩余牌)
        for i, opp in enumerate([p for p in range(6) if p % 2 != team]):
            s[39+i] = self.hands[opp].sum() / 10.0
        
        # 是否轮到自己 (1)
        s[42] = 1.0 if player == self.current else 0.0
        
        # 是否首发 (1)
        s[43] = 1.0 if self.last_play is None else 0.0
        
        # 其他位置编码 (填充到120维)
        s[44:50] = [1.0 if self.finished[i] else 0.0 for i in range(6)]
        
        return s
    
    def get_actions(self):
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
            # 首发：可以选择任意牌型
            for i in range(15):
                if hand[i] >= 1:
                    actions.append((i, 1))
            for i in range(13):
                if hand[i] >= 2:
                    actions.append((i, 2))
            for i in range(13):
                if hand[i] >= 3:
                    actions.append((i, 3))
            # 添加对子、三张的优先级
            actions.sort(key=lambda x: x[0])  # 按牌值排序
        else:
            # 跟牌：必须出更大的同类型牌
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            
            for i in range(last_val + 1, 15):
                if hand[i] >= last_cnt:
                    actions.append((i, last_cnt))
            actions.append((-1, 0))  # 过牌
        
        return actions if actions else [(-1, 0)]
    
    def step(self, card, count):
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
            self.cards_played[self.current % 2] += count
        
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        
        # 检查出完 - 游戏继续直到所有6人都出完
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            self.finish_order.append(self.current)
            
            # 只有全部6人都出完才结束
            if all(self.finished):
                return True, self.finish_order[0] % 2  # 返回头游所在队伍
        
        if action.sum() > 0:
            self.last_play = action.copy()
            self.last_player = self.current
            self.passes = 0
        else:
            self.passes += 1
        
        if self.passes >= 5:
            self.last_play = None
            self.last_player = -1
            self.passes = 0
        
        self.current = (self.current + 1) % 6
        while self.finished[self.current]:
            self.current = (self.current + 1) % 6
        
        return False, -1


# ============== 改进的网络结构 ==============
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(120, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 15)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)


class ValueNet(nn.Module):
    """价值网络 - 估计局面优势"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(120, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


# ============== 动作选择 ==============
def model_action(policy_net, state, actions, epsilon=0.1):
    """模型决策，带探索"""
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    
    with torch.no_grad():
        probs = policy_net(torch.FloatTensor(state))
    
    # 综合考虑概率和牌值
    scores = []
    for i, (card, count) in enumerate(actions):
        if card >= 0:
            # 大牌给更高权重
            score = probs[card].item() * (1 + card * 0.1) * count
            scores.append((i, score))
        else:
            scores.append((i, 0.001))  # 过牌低概率
    
    return max(scores, key=lambda x: x[1])[0]


def random_action(actions):
    """随机策略"""
    return np.random.randint(len(actions))


def rule_action(actions):
    """规则Bot：智能小牌优先"""
    valid = [(i, card, count) for i, (card, count) in enumerate(actions) if card >= 0]
    
    if not valid:
        return len(actions) - 1  # 过牌
    
    # 优先出对子和三张（快速消耗手牌）
    pairs = [(i, card, count) for i, card, count in valid if count >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])  # 小牌优先
        return pairs[0][0]
    
    # 单张
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


# ============== 训练函数 ==============
def play_game(policy_net, opponent_type='mixed', epsilon=0.15):
    """进行一局训练
    opponent_type: 'random', 'rule', 'mixed'
    """
    game = FastGame()
    game.reset()
    
    model_states = []
    model_actions = []
    model_players = []
    
    while True:
        player = game.current
        actions = game.get_actions()
        state = game.get_state(player)
        
        if player % 2 == 0:  # 红队：模型
            idx = model_action(policy_net, state, actions, epsilon)
            model_states.append(state)
            model_actions.append(actions[idx][0])
            model_players.append(player)
        else:  # 蓝队：对手
            if opponent_type == 'mixed':
                # 随机混合对手
                if np.random.random() < 0.3:
                    idx = random_action(actions)
                else:
                    idx = rule_action(actions)
            elif opponent_type == 'random':
                idx = random_action(actions)
            else:
                idx = rule_action(actions)
        
        card, count = actions[idx]
        done, winner = game.step(card, count)
        
        if done:
            return model_states, model_actions, model_players, winner, game.finish_order


def compute_advantage(finish_order, model_players):
    """计算优势奖励"""
    if not finish_order:
        return 0
    
    # 头游奖励
    first = finish_order[0]
    first_team = first % 2
    
    # 计算红队玩家在出完顺序中的位置
    red_positions = [i for i, p in enumerate(finish_order) if p % 2 == 0]
    blue_positions = [i for i, p in enumerate(finish_order) if p % 2 == 1]
    
    # 位置越靠前越好
    advantage = 0
    if red_positions:
        advantage += (5 - red_positions[0]) * 0.1  # 头游奖励
    if blue_positions:
        advantage -= (5 - blue_positions[0]) * 0.1  # 对手头游惩罚
    
    return advantage


def train_step(policy_net, optimizer, states, actions, players, winner, finish_order):
    """单步训练"""
    # 基础奖励
    if winner == 0:  # 红队胜
        base_reward = 1.0
    else:
        base_reward = -0.5
    
    # 优势奖励
    advantage = compute_advantage(finish_order, players)
    
    total_loss = 0
    for i, (state, action, player) in enumerate(zip(states, actions, players)):
        # 根据出牌顺序调整奖励
        position_bonus = 0
        if player in finish_order:
            pos = finish_order.index(player)
            position_bonus = (len(finish_order) - pos) * 0.1
        
        reward = base_reward + advantage + position_bonus
        
        probs = policy_net(torch.FloatTensor(state))
        
        if 0 <= action < 15:
            prob = probs[action]
        else:
            prob = probs.mean()
        
        loss = -torch.log(prob + 1e-8) * reward
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss


def test(policy_net, num=100, opponent='rule'):
    """测试胜率"""
    wins = 0
    for _ in range(num):
        _, _, _, winner, _ = play_game(policy_net, opponent, epsilon=0.0)
        if winner == 0:
            wins += 1
    return wins / num * 100


def main():
    print("=" * 70)
    print("强化学习训练 V4 - 改进版")
    print("改进：混合对手 + 更好奖励 + 更强网络")
    print("=" * 70)
    
    policy_net = PolicyNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    # 加载已有模型
    model_path = '/workspace/projects/rl_v4.pt'
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print(f"✅ 加载模型: {model_path}")
    
    target = 100000
    test_interval = 1000
    save_interval = 5000
    
    # 统计
    total_wins = 0
    total_games = 0
    total_scores = 0  # 得分统计
    recent_wins = deque(maxlen=100)
    recent_scores = deque(maxlen=100)  # 最近得分情况
    best_rate = 0
    start_time = time.time()
    
    print(f"\n开始训练: 0 -> {target:,} 局")
    print("-" * 70)
    print(f"{'局数':>10} | {'vs规则':>8} | {'vs随机':>8} | {'最佳':>8} | {'速度':>10}")
    print("-" * 70)
    
    for ep in range(target):
        # 训练阶段用混合对手
        states, actions, players, winner, finish_order = play_game(
            policy_net, opponent_type='mixed', epsilon=0.2
        )
        
        train_step(policy_net, optimizer, states, actions, players, winner, finish_order)
        
        total_games += 1
        if winner == 0:
            total_wins += 1
            recent_wins.append(1)
        else:
            recent_wins.append(0)
        
        # 计算得分：红队获胜（头游）且最后一名是蓝队
        if winner == 0:  # 红队获胜（头游是红队）
            # 检查最后一名是否是蓝队
            if finish_order and finish_order[-1] % 2 == 1:  # 最后一名是蓝队
                total_scores += 1
                recent_scores.append(1)
            else:
                recent_scores.append(0)
        else:
            recent_scores.append(0)
        
        # 测试
        if (ep + 1) % test_interval == 0:
            rate_rule = test(policy_net, 50, 'rule')
            rate_random = test(policy_net, 50, 'random')
            
            elapsed = time.time() - start_time
            speed = (ep + 1) / elapsed
            
            train_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
            total_win_rate = total_wins / total_games * 100 if total_games > 0 else 0
            score_rate = sum(recent_scores) / len(recent_scores) * 100 if recent_scores else 0
            
            best = max(rate_rule, best_rate)
            if rate_rule > best_rate:
                best_rate = rate_rule
                torch.save(policy_net.state_dict(), '/workspace/projects/rl_v4_best.pt')
            
            print(f"{ep+1:>10,} | {rate_rule:>7.1f}% | {rate_random:>7.1f}% | {best_rate:>7.1f}% | {speed:>8.0f}局/秒")
            
            # 保存状态
            status = {
                'episode': ep + 1,
                'target': target,
                'progress': (ep + 1) / target * 100,
                'win_rate': round(train_rate, 1),
                'total_win_rate': round(total_win_rate, 1),
                'score_rate': round(score_rate, 1),
                'vs_rule': round(rate_rule, 1),
                'vs_random': round(rate_random, 1),
                'best_rate': round(best_rate, 1),
                'total_wins': total_wins,
                'total_games': total_games,
                'speed': round(speed, 1),
                'elapsed_hours': round(elapsed / 3600, 2),
                'training_type': 'reinforcement_learning_v4',
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
        
        # 保存模型
        if (ep + 1) % save_interval == 0:
            torch.save(policy_net.state_dict(), model_path)
    
    # 训练完成
    print("-" * 70)
    print(f"训练完成！最佳胜率: {best_rate:.1f}%")
    torch.save(policy_net.state_dict(), '/workspace/projects/rl_v4_final.pt')
    
    # 最终测试
    final_rule = test(policy_net, 100, 'rule')
    final_random = test(policy_net, 100, 'random')
    print(f"最终测试: vs规则Bot {final_rule:.1f}%, vs随机Bot {final_random:.1f}%")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
训练 V8 - Actor-Critic + 价值网络 + 不对称训练
改进：
1. 价值网络预测局面胜率，提供中间奖励
2. 不对称训练：对手使用次优策略
3. Actor-Critic架构，更稳定的训练
4. 增加训练局数到50万
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from datetime import datetime
from collections import deque
import random

# ============== 游戏环境 ==============
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        self.history = []  # 出牌历史
        
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
        self.history = []
    
    def get_state(self):
        """丰富的状态表示 (100维)"""
        s = np.zeros(100, dtype=np.float32)
        
        # 自己手牌 (15)
        s[:15] = self.hands[self.current] / 4.0
        
        # 上一轮出牌 (15)
        if self.last_play is not None:
            s[15:30] = self.last_play / 4.0
        
        # 所有玩家剩余牌数 (6)
        for i in range(6):
            s[30+i] = self.hands[i].sum() / 10.0
        
        # 出牌类型 (3)
        if self.last_play is not None and self.last_play.sum() > 0:
            cnt = int(self.last_play[self.last_play > 0][0])
            s[36 + min(cnt-1, 2)] = 1.0
        
        # 是否首发 (1)
        s[39] = 1.0 if self.last_play is None else 0.0
        
        team = self.current % 2
        
        # 队友状态 (6)
        for i, p in enumerate([p for p in range(6) if p % 2 == team and p != self.current]):
            s[40+i] = 1.0 if self.finished[p] else 0.0
            s[46+i] = self.hands[p].sum() / 10.0
        
        # 对手状态 (6)
        for i, p in enumerate([p for p in range(6) if p % 2 != team]):
            s[43+i] = 1.0 if self.finished[p] else 0.0
            s[49+i] = self.hands[p].sum() / 10.0
        
        # 手牌统计 (10)
        hand = self.hands[self.current]
        s[52] = sum(1 for i in range(13) if hand[i] == 1) / 13.0
        s[53] = sum(1 for i in range(13) if hand[i] == 2) / 6.0
        s[54] = sum(1 for i in range(13) if hand[i] >= 3) / 4.0
        s[55] = hand[13] / 2.0
        s[56] = hand[14] / 2.0
        s[57] = hand.sum() / 10.0
        s[58] = sum(1 for p in range(6) if p % 2 == team and self.finished[p]) / 3.0
        s[59] = sum(1 for p in range(6) if p % 2 != team and self.finished[p]) / 3.0
        
        # 头游信息 (1)
        if self.finish_order:
            s[60] = 1.0 if self.finish_order[0] % 2 == team else -1.0
        
        # 关键牌 (3)
        s[61] = 1.0 if hand[13] > 0 else 0.0
        s[62] = 1.0 if hand[14] > 0 else 0.0
        s[63] = hand[:13].max() / 3.0 if hand[:13].max() > 0 else 0
        
        # 回合数 (1)
        s[64] = min(len(self.history) / 100.0, 1.0)
        
        # 红队已出完数 (1)
        s[65] = sum(1 for p in range(6) if p % 2 == 0 and self.finished[p]) / 3.0
        # 蓝队已出完数 (1)
        s[66] = sum(1 for p in range(6) if p % 2 == 1 and self.finished[p]) / 3.0
        
        # 剩余总牌数 (1)
        s[67] = self.hands.sum() / 50.0
        
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
    
    def step(self, card, count):
        action = np.zeros(15, dtype=np.int32)
        if card >= 0 and count > 0:
            action[card] = count
            self.history.append((self.current, card, count))
        
        self.hands[self.current] -= action
        self.hands = np.maximum(self.hands, 0)
        
        if self.hands[self.current].sum() == 0:
            self.finished[self.current] = True
            self.finish_order.append(self.current)
            if all(self.finished):
                return True, self.finish_order[0] % 2
        
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


# ============== Actor-Critic 网络 ==============
class ActorCritic(nn.Module):
    def __init__(self, state_dim=100, hidden_dim=512):
        super().__init__()
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor: 策略网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 16)  # 15种牌 + 过牌
        )
        
        # Critic: 价值网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # 输出 [-1, 1]
        )
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, state, actions):
        with torch.no_grad():
            logits, _ = self(torch.FloatTensor(state))
        valid_idx = [15 if c < 0 else c for c, _ in actions]
        probs = torch.softmax(logits, dim=0)
        scores = [(i, probs[idx].item()) for i, idx in enumerate(valid_idx)]
        return max(scores, key=lambda x: x[1])[0]
    
    def get_value(self, state):
        with torch.no_grad():
            _, value = self(torch.FloatTensor(state))
        return value.item()


# ============== 对手策略 ==============
def rule_action(game):
    """普通规则Bot"""
    actions = game.get_actions()
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


def weak_rule_action(game, noise=0.3):
    """次优规则Bot - 有概率犯错"""
    if np.random.random() < noise:
        actions = game.get_actions()
        return np.random.randint(len(actions))
    return rule_action(game)


def smart_action(game):
    """智能规则Bot"""
    actions = game.get_actions()
    hand = game.hands[game.current]
    team = game.current % 2
    
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    opps_remaining = [game.hands[p].sum() for p in range(6) if p % 2 != team and not game.finished[p]]
    
    valid = [(i, c, cnt) for i, (c, cnt) in enumerate(actions) if c >= 0]
    if not valid: return len(actions) - 1
    
    # 策略调整
    if mates_done >= 2:
        valid.sort(key=lambda x: -x[1])  # 大牌优先
        return valid[0][0]
    
    if opps_remaining and min(opps_remaining) <= 3:
        valid.sort(key=lambda x: -x[1])
        pairs = [x for x in valid if x[2] >= 2]
        if pairs: return pairs[0][0]
        return valid[0][0]
    
    if hand.sum() <= 4:
        valid.sort(key=lambda x: -x[1])
        return valid[0][0]
    
    pairs = [(i, c, cnt) for i, c, cnt in valid if cnt >= 2]
    if pairs:
        pairs.sort(key=lambda x: x[1])
        return pairs[0][0]
    
    valid.sort(key=lambda x: x[1])
    return valid[0][0]


# ============== 训练函数 ==============
def compute_advantage(model, game):
    """计算当前局面的优势"""
    state = game.get_state()
    value = model.get_value(state)
    
    # 额外奖励：头游、队友领先等
    team = game.current % 2
    bonus = 0.0
    
    if game.finish_order:
        if game.finish_order[0] % 2 == team:
            bonus += 0.3  # 头游优势
    
    mates_done = sum(1 for p in range(6) if p % 2 == team and game.finished[p])
    opps_done = sum(1 for p in range(6) if p % 2 != team and game.finished[p])
    bonus += (mates_done - opps_done) * 0.1
    
    return value + bonus


def train_episode(model, optimizer, opponent_type='weak', gamma=0.99):
    """训练一局游戏"""
    game = Game()
    game.reset()
    
    states, actions, rewards, values = [], [], [], []
    
    while True:
        player = game.current
        state = game.get_state()
        action_list = game.get_actions()
        
        if player % 2 == 0:  # 红队：模型
            logits, value = model(torch.FloatTensor(state))
            
            # 从合法动作中选择
            valid_idx = [15 if c < 0 else c for c, cnt in action_list]
            probs = torch.softmax(logits, dim=0)
            
            # 安全地获取合法动作的概率
            action_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid_idx])
            action_probs = action_probs / action_probs.sum()  # 归一化
            
            if np.random.random() < 0.1:  # 10%探索
                action_idx = np.random.randint(len(valid_idx))
            else:
                action_idx = torch.multinomial(action_probs, 1).item()
            
            card, cnt = action_list[action_idx]
            action_code = valid_idx[action_idx]
            
            states.append(state)
            actions.append(action_code)
            values.append(value.item())
        else:  # 蓝队：对手
            if opponent_type == 'weak':
                idx = weak_rule_action(game, noise=0.4)
            elif opponent_type == 'smart':
                idx = smart_action(game)
            elif opponent_type == 'random':
                idx = np.random.randint(len(action_list))
            else:
                idx = rule_action(game)
            card, cnt = action_list[idx]
        
        done, winner = game.step(card, cnt)
        
        # 计算奖励
        if done:
            final_reward = 1.0 if winner == 0 else -0.5
            if winner == 0 and game.finish_order and game.finish_order[-1] % 2 == 1:
                final_reward += 0.3  # 得分奖励
            rewards = [final_reward] * len(states)
        
        if done:
            break
    
    # 计算优势并更新
    if not states:
        return 0.0, winner
    
    # Actor-Critic更新
    total_loss = 0.0
    returns = []
    R = rewards[-1]
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.FloatTensor(returns)
    values_t = torch.FloatTensor(values)
    advantages = returns - values_t
    
    for i, (state, action, advantage, ret) in enumerate(zip(states, actions, advantages, returns)):
        logits, value = model(torch.FloatTensor(state))
        
        # Critic loss
        critic_loss = (value - ret).pow(2)
        
        # Actor loss
        probs = torch.softmax(logits, dim=0)
        if action < len(probs):
            actor_loss = -torch.log(probs[action] + 1e-8) * advantage
        else:
            actor_loss = torch.tensor(0.0)
        
        loss = critic_loss + actor_loss
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
    return total_loss, winner


def test(model, n=200, opp='rule'):
    """测试模型"""
    game = Game()
    wins, scores = 0, 0
    
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx = model.get_action(game.get_state(), actions)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                elif opp == 'smart':
                    idx = smart_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner = game.step(*actions[idx])
            if done:
                if winner == 0:
                    wins += 1
                    if game.finish_order and game.finish_order[-1] % 2 == 1:
                        scores += 1
                break
    
    return wins / n * 100, scores / n * 100


def main():
    print("=" * 60)
    print("V8 Actor-Critic + 价值网络 + 不对称训练")
    print("=" * 60)
    
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 加载预训练模型
    if os.path.exists('/workspace/projects/rl_v7_final.pt'):
        try:
            # 尝试加载兼容的权重
            old_state = torch.load('/workspace/projects/rl_v7_final.pt', weights_only=True)
            # 只加载能匹配的权重
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in old_state.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict, strict=False)
            print("✅ 加载预训练权重")
        except:
            print("⚠️ 无法加载预训练权重，从头训练")
    
    target = 100000  # 10万局
    test_interval = 2000
    save_interval = 10000
    
    wins_history = deque(maxlen=200)
    best_rule = 0
    best_random = 0
    start_time = datetime.now()
    
    print(f"\n开始训练: 0 -> {target:,} 局")
    print("-" * 60)
    print(f"{'局数':>8} | {'vs规则':>7} | {'vs随机':>7} | {'vs智能':>7} | {'最佳':>7} | {'对手':>8}")
    print("-" * 60)
    
    for ep in range(target):
        # 渐进式对手：从弱到强
        progress = ep / target
        if progress < 0.2:
            opp_type = 'random'  # 20%：随机对手
        elif progress < 0.5:
            opp_type = 'weak'    # 30%：弱规则对手
        elif progress < 0.8:
            opp_type = 'rule'    # 30%：普通规则对手
        else:
            opp_type = 'smart'   # 20%：智能对手
        
        loss, winner = train_episode(model, optimizer, opp_type)
        wins_history.append(1 if winner == 0 else 0)
        
        # 测试
        if (ep + 1) % test_interval == 0:
            r1, s1 = test(model, 200, 'rule')
            r2, s2 = test(model, 200, 'random')
            r3, s3 = test(model, 200, 'smart')
            
            if r1 > best_rule:
                best_rule = r1
                torch.save(model.state_dict(), '/workspace/projects/rl_v8_best_rule.pt')
            if r2 > best_random:
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v8_best_random.pt')
            
            train_rate = sum(wins_history) / len(wins_history) * 100
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            
            print(f"{ep+1:>8,} | {r1:>6.1f}% | {r2:>6.1f}% | {r3:>6.1f}% | {max(best_rule, best_random):>6.1f}% | {opp_type:>8}")
            
            # 保存状态
            status = {
                'episode': ep + 1,
                'target': target,
                'progress': (ep + 1) / target * 100,
                'vs_rule': round(r1, 1),
                'vs_random': round(r2, 1),
                'vs_smart': round(r3, 1),
                'best_rule': round(best_rule, 1),
                'best_random': round(best_random, 1),
                'train_rate': round(train_rate, 1),
                'elapsed_hours': round(elapsed, 2),
                'opponent': opp_type,
                'training_type': 'v8_actor_critic',
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            with open('/workspace/projects/training_status.json', 'w') as f:
                json.dump(status, f, indent=2)
        
        # 保存模型
        if (ep + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'/workspace/projects/rl_v8_ep{ep+1}.pt')
    
    # 最终测试
    print("-" * 60)
    print("\n最终测试 (1000局):")
    
    model.load_state_dict(torch.load('/workspace/projects/rl_v8_best_rule.pt', weights_only=True))
    r1, s1 = test(model, 1000, 'rule')
    r2, s2 = test(model, 1000, 'random')
    r3, s3 = test(model, 1000, 'smart')
    
    print(f"  vs规则Bot: {r1:.1f}% (得分率: {s1:.1f}%)")
    print(f"  vs随机Bot: {r2:.1f}% (得分率: {s2:.1f}%)")
    print(f"  vs智能Bot: {r3:.1f}% (得分率: {s3:.1f}%)")
    
    torch.save(model.state_dict(), '/workspace/projects/rl_v8_final.pt')
    
    # 判断是否达标
    print("\n" + "=" * 60)
    if r1 >= 60:
        print("✅ vs规则Bot: 达标")
    else:
        print(f"⚠️ vs规则Bot: 未达标 ({r1:.1f}% < 60%)")
    if r2 >= 60:
        print("✅ vs随机Bot: 达标")
    else:
        print(f"⚠️ vs随机Bot: 未达标 ({r2:.1f}% < 60%)")
    
    # 更新状态
    status['status'] = 'completed'
    status['final_vs_rule'] = round(r1, 1)
    status['final_vs_random'] = round(r2, 1)
    with open('/workspace/projects/training_status.json', 'w') as f:
        json.dump(status, f, indent=2)


if __name__ == '__main__':
    main()

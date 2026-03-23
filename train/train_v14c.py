#!/usr/bin/env python3
"""
V14c - 四带一策略优化版
基于V14b，新增四带一优先带小单张策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import functools
print = functools.partial(print, flush=True)


# ============== 增强游戏环境（支持四带一） ==============
class Game:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)  # 0-12: 2-A, 13: 小王, 14: 大王
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.trick_count = 0
        self.last_play_type = None  # 记录牌型
        
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
        self.last_play_type = None
        return self.get_state()
    
    def get_state(self):
        s = np.zeros(60, dtype=np.float32)
        s[:15] = self.hands[self.current]
        if self.last_play is not None:
            s[15:30] = self.last_play
        for i in range(6):
            s[30+i] = self.hands[i].sum()
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
        s[51] = hand.sum()
        return s
    
    def get_actions(self):
        """获取合法动作，支持单张、对子、三张、四带一"""
        hand = self.hands[self.current]
        actions = []
        
        if self.last_play is None or self.last_player == self.current:
            # 首发：可以出任意牌型
            # 单张
            for i in range(15):
                if hand[i] >= 1:
                    actions.append((i, 1, 'single'))
            # 对子
            for i in range(13):
                if hand[i] >= 2:
                    actions.append((i, 2, 'pair'))
            # 三张
            for i in range(13):
                if hand[i] >= 3:
                    actions.append((i, 3, 'triple'))
            # 四带一（需要4张相同 + 1张单牌）
            for i in range(13):
                if hand[i] >= 4:
                    # 找可以带的单牌
                    for j in range(15):
                        if j != i and hand[j] >= 1:
                            actions.append((i, 4, 'four_with_one', j))  # j是带的单牌
        else:
            # 跟牌：需要匹配牌型
            last_val = int(self.last_play.max()) if self.last_play.max() > 0 else -1
            last_cnt = int(self.last_play[self.last_play > 0][0]) if self.last_play.sum() > 0 else 1
            
            # 根据上家牌型决定跟牌
            if self.last_play_type == 'four_with_one':
                # 四带一：需要更大的四带一
                for i in range(last_val + 1, 13):
                    if hand[i] >= 4:
                        for j in range(15):
                            if j != i and hand[j] >= 1:
                                actions.append((i, 4, 'four_with_one', j))
            elif last_cnt == 4:
                # 四张（可能需要跟四带一）
                for i in range(last_val + 1, 15):
                    if hand[i] >= 4:
                        for j in range(15):
                            if j != i and hand[j] >= 1:
                                actions.append((i, 4, 'four_with_one', j))
            else:
                # 普通跟牌
                for i in range(last_val + 1, 15):
                    if hand[i] >= last_cnt:
                        if last_cnt == 1:
                            actions.append((i, 1, 'single'))
                        elif last_cnt == 2:
                            actions.append((i, 2, 'pair'))
                        elif last_cnt == 3:
                            actions.append((i, 3, 'triple'))
            
            # 可以pass
            actions.append((-1, 0, 'pass'))
        
        return actions if actions else [(-1, 0, 'pass')]
    
    def step(self, action_tuple):
        """执行动作并返回奖励"""
        reward = 0.0
        strategic_reward = 0.0
        
        hand = self.hands[self.current]
        team = self.current % 2
        
        if len(action_tuple) == 4:
            # 四带一: (主牌, 4, 'four_with_one', 带的单牌)
            main_card, cnt, play_type, single_card = action_tuple
        elif len(action_tuple) == 3:
            main_card, cnt, play_type = action_tuple
            single_card = None
        else:
            main_card, cnt = action_tuple[0], action_tuple[1]
            play_type = 'single' if cnt == 1 else 'pair' if cnt == 2 else 'triple'
            single_card = None
        
        if main_card >= 0:
            # 出牌
            hand[main_card] -= cnt
            if single_card is not None:
                hand[single_card] -= 1
            
            # 基础奖励
            total_cards = cnt + (1 if single_card else 0)
            reward += 0.1 + total_cards * 0.05
            
            # ====== 四带一策略奖励 ======
            if play_type == 'four_with_one' and single_card is not None:
                # 四带一：优先带小单张
                if single_card <= 5:  # 小单张 (2-7)
                    strategic_reward += 0.15  # 带小牌奖励
                elif single_card <= 8:  # 中等单张
                    strategic_reward += 0.08
                elif single_card >= 11:  # 大单张 (J, Q, K, A)
                    strategic_reward -= 0.1  # 带大牌惩罚
                
                # 带单张（而不是带对子拆牌）额外奖励
                if hand[single_card] == 0:  # 单张刚好用完
                    strategic_reward += 0.05
            
            # ====== 三带二策略奖励 ======
            if cnt == 3:
                # 找带的对子（简化：假设带了最小的对子）
                pairs = [i for i in range(13) if hand[i] >= 2 and i != main_card]
                if pairs:
                    min_pair = min(pairs)
                    if min_pair <= 5:
                        strategic_reward += 0.1  # 带小对子
                    elif min_pair >= 10:
                        strategic_reward -= 0.05  # 带大对子惩罚
            
            # ====== 首发策略 ======
            if self.last_play is None or self.last_player == self.current:
                if main_card <= 5:
                    strategic_reward += 0.05
                elif main_card <= 8:
                    strategic_reward += 0.02
                elif main_card >= 11 and cnt == 1:
                    strategic_reward -= 0.03
            
            # ====== 大小王策略 ======
            if main_card == 14:  # 大王
                if self.trick_count < 3:
                    strategic_reward -= 0.2
                elif self.trick_count < 6:
                    strategic_reward -= 0.1
                else:
                    strategic_reward += 0.1
            elif main_card == 13:  # 小王
                if self.trick_count < 2:
                    strategic_reward -= 0.15
                elif self.trick_count < 5:
                    strategic_reward -= 0.05
                else:
                    strategic_reward += 0.05
            
            # 四带一用王作为单牌
            if single_card == 14:
                if self.trick_count < 3:
                    strategic_reward -= 0.15
            elif single_card == 13:
                if self.trick_count < 2:
                    strategic_reward -= 0.1
            
            # ====== 配合策略 ======
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None and self.last_play.max() > 0 else 0
                if last_val >= 11:
                    strategic_reward -= 0.15
                elif last_val >= 9:
                    strategic_reward -= 0.05
            
            # 更新游戏状态
            self.last_play = np.zeros(15, dtype=np.int32)
            self.last_play[main_card] = cnt
            if single_card is not None:
                self.last_play[single_card] = 1
            self.last_player = self.current
            self.last_play_type = play_type
            self.passes = 0
            
            if hand.sum() == 0:
                self.finished[self.current] = True
                reward += 0.5
        else:
            # Pass
            reward -= 0.02
            
            if self.last_player >= 0 and self.last_player % 2 == team:
                last_val = int(self.last_play.max()) if self.last_play is not None and self.last_play.max() > 0 else 0
                if last_val >= 10:
                    strategic_reward += 0.1
            
            self.passes += 1
        
        total_reward = reward + strategic_reward
        
        # 下一个玩家
        for _ in range(6):
            self.current = (self.current + 1) % 6
            if not self.finished[self.current]:
                break
        
        if self.passes >= sum(1 for p in range(6) if not self.finished[p]):
            self.last_play = None
            self.last_player = -1
            self.passes = 0
            self.trick_count += 1
            self.last_play_type = None
        
        team0_done = all(self.finished[p] for p in [0, 2, 4])
        team1_done = all(self.finished[p] for p in [1, 3, 5])
        
        if team0_done or team1_done:
            winner = 0 if team0_done else 1
            if winner == 0:
                total_reward += 2.0
            return True, winner, total_reward
        
        return False, -1, total_reward


# ============== 网络结构 ==============
class ActorCritic(nn.Module):
    def __init__(self, state_dim=60, action_dim=16, hidden_dim=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, state, actions, deterministic=True):
        with torch.no_grad():
            logits, value = self(torch.FloatTensor(state))
        # 简化：只考虑主牌值
        valid = [15 if a[0] < 0 else a[0] for a in actions]
        probs = torch.softmax(logits, 0)
        valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
        valid_probs = valid_probs / valid_probs.sum()
        if deterministic:
            a_idx = valid_probs.argmax().item()
        else:
            a_idx = torch.multinomial(valid_probs, 1).item()
        return a_idx, valid[a_idx], value.item()


# ============== 规则AI（支持四带一） ==============
def rule_action(game):
    actions = game.get_actions()
    
    # 优先四带一带小牌
    four_with_one = [a for a in actions if len(a) >= 3 and a[2] == 'four_with_one']
    if four_with_one:
        # 按带的单牌排序，优先带小牌
        four_with_one.sort(key=lambda x: x[3] if len(x) > 3 else 100)
        # 选择带最小单牌的四带一
        best = four_with_one[0]
        return actions.index(best)
    
    # 对子
    pairs = [(i, a) for i, a in enumerate(actions) if len(a) >= 3 and a[2] == 'pair' and a[0] >= 0]
    if pairs:
        pairs.sort(key=lambda x: x[1][0])  # 按牌值排序
        return pairs[0][0]
    
    # 单张
    singles = [(i, a) for i, a in enumerate(actions) if len(a) >= 3 and a[2] == 'single' and a[0] >= 0]
    if singles:
        singles.sort(key=lambda x: x[1][0])
        return singles[0][0]
    
    # pass
    return len(actions) - 1


def test(model, n=500, opp='rule'):
    game = Game()
    wins = 0
    for _ in range(n):
        game.reset()
        while True:
            actions = game.get_actions()
            if game.current % 2 == 0:
                idx, _, _ = model.get_action(game.get_state(), actions, deterministic=True)
            else:
                if opp == 'rule':
                    idx = rule_action(game)
                else:
                    idx = np.random.randint(len(actions))
            done, winner, _ = game.step(actions[idx])
            if done:
                if winner == 0: wins += 1
                break
    return wins / n * 100


# ============== 训练 ==============
def finetune(model, n_games=5000, lr=1e-4):
    game = Game()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_rule = 0
    best_random = 0
    best_combined = 0
    
    memory = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': []}
    
    print(f"\n策略微调 ({n_games}局)...")
    print("=" * 60)
    
    for g in range(n_games):
        game.reset()
        episode_data = []
        
        while True:
            state = game.get_state()
            actions = game.get_actions()
            
            if game.current % 2 == 0:
                idx, action, value = model.get_action(state, actions, deterministic=False)
                with torch.no_grad():
                    logits, _ = model(torch.FloatTensor(state))
                log_prob = torch.log_softmax(logits, 0)[action].item()
                episode_data.append({'state': state, 'action': action, 'value': value, 'log_prob': log_prob})
                action_tuple = actions[idx]
            else:
                actions_opp = game.get_actions()
                if np.random.random() < 0.5:
                    idx = np.random.randint(len(actions_opp))
                else:
                    idx = rule_action(game)
                action_tuple = actions_opp[idx]
            
            done, winner, reward = game.step(action_tuple)
            if episode_data:
                episode_data[-1]['reward'] = reward
            
            if done:
                returns = []
                R = 0
                for data in reversed(episode_data):
                    R = data.get('reward', 0) + 0.99 * R
                    returns.insert(0, R)
                
                for i, data in enumerate(episode_data):
                    memory['states'].append(data['state'])
                    memory['actions'].append(data['action'])
                    memory['rewards'].append(returns[i])
                    memory['values'].append(data['value'])
                    memory['log_probs'].append(data['log_prob'])
                break
        
        if len(memory['states']) >= 1024:
            states = torch.FloatTensor(np.array(memory['states']))
            actions = torch.LongTensor(memory['actions'])
            old_log_probs = torch.FloatTensor(memory['log_probs'])
            returns = torch.FloatTensor(memory['rewards'])
            old_values = torch.FloatTensor(memory['values'])
            
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(2):
                logits, new_values = model(states)
                new_log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values.squeeze(), returns)
                entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            memory = {k: [] for k in memory}
        
        if (g + 1) % 500 == 0:
            r1 = test(model, 400, 'rule')
            r2 = test(model, 400, 'random')
            combined = r1 + r2
            
            print(f"  {g+1:>5,}局 | vs规则: {r1:.1f}% | vs随机: {r2:.1f}%")
            
            if combined > best_combined:
                best_combined = combined
                best_rule = r1
                best_random = r2
                torch.save(model.state_dict(), '/workspace/projects/rl_v14c_best.pt')
                print(f"    ★ 新最佳! 综合: {combined:.1f}%")
    
    return model, best_rule, best_random


def main():
    print("=" * 60)
    print("V14c - 四带一策略优化版")
    print("优化：四带一优先带小单张、三带二、配合度、大小王")
    print("=" * 60)
    
    model = ActorCritic(hidden_dim=512)
    
    # 加载V14b模型
    if os.path.exists('/workspace/projects/rl_v14b_best.pt'):
        print("加载V14b模型...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v14b_best.pt', weights_only=True))
    elif os.path.exists('/workspace/projects/rl_v13_ppo_best.pt'):
        print("加载V13 PPO模型...")
        model.load_state_dict(torch.load('/workspace/projects/rl_v13_ppo_best.pt', weights_only=True))
    
    # 初始测试
    r1 = test(model, 500, 'rule')
    r2 = test(model, 500, 'random')
    print(f"初始: vs规则 {r1:.1f}%, vs随机 {r2:.1f}%")
    
    # 微调
    model, best_rule, best_random = finetune(model, n_games=5000)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试:")
    
    if os.path.exists('/workspace/projects/rl_v14c_best.pt'):
        model.load_state_dict(torch.load('/workspace/projects/rl_v14c_best.pt', weights_only=True))
    
    r1 = test(model, 1000, 'rule')
    r2 = test(model, 1000, 'random')
    
    print(f"  vs规则Bot: {r1:.1f}%")
    print(f"  vs随机Bot: {r2:.1f}%")
    
    print("\n" + "=" * 60)
    print("V14c微调完成!")
    print(f"最佳: vs规则 {max(r1, best_rule):.1f}%, vs随机 {max(r2, best_random):.1f}%")


if __name__ == '__main__':
    main()

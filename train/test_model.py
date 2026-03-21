#!/usr/bin/env python3
"""测试模型对规则Bot的胜率"""
import torch
import torch.nn as nn
import numpy as np
import json

# 复用训练脚本中的类
class FastGame:
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        self.finish_order = []
        
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
        
    def get_state(self, player):
        state = np.zeros(90, dtype=np.float32)
        state[:15] = self.hands[player] / 4.0
        for i, p in enumerate([(player+1)%6, (player+3)%6, (player+5)%6]):
            state[15+i*5] = self.hands[p].sum() / 9.0
        if self.last_play is not None:
            state[75:90] = self.last_play / 4.0
        return state
    
    def get_actions(self, player):
        hand = self.hands[player]
        actions = []
        if self.last_play is None or self.last_player == player:
            for i in range(15):
                if hand[i] >= 1:
                    a = np.zeros(15, dtype=np.int32); a[i] = 1; actions.append(a)
            for i in range(13):
                if hand[i] >= 2:
                    a = np.zeros(15, dtype=np.int32); a[i] = 2; actions.append(a)
            for i in range(13):
                if hand[i] >= 3:
                    a = np.zeros(15, dtype=np.int32); a[i] = 3; actions.append(a)
        else:
            last = self.last_play
            last_val = last.max()
            last_cnt = int(last[last > 0][0]) if len(last[last > 0]) > 0 else 0
            for i in range(int(last_val) + 1, 15):
                if hand[i] >= last_cnt:
                    a = np.zeros(15, dtype=np.int32); a[i] = last_cnt; actions.append(a)
            if last_cnt < 4:
                for i in range(13):
                    if hand[i] >= 4:
                        a = np.zeros(15, dtype=np.int32); a[i] = 4; actions.append(a)
            actions.append(np.zeros(15, dtype=np.int32))
        return actions if actions else [np.zeros(15, dtype=np.int32)]
    
    def step(self, player, action):
        self.hands[player] -= action
        self.hands = np.maximum(self.hands, 0)
        if self.hands[player].sum() == 0:
            self.finished[player] = True
            self.finish_order.append(player)
            team = player % 2
            if all(self.finished[i] for i in range(6) if i % 2 == team):
                return team
        if action.sum() > 0:
            self.last_play = action.copy()
            self.last_player = player
            self.passes = 0
        else:
            self.passes += 1
        if self.passes >= 5:
            self.last_play = None
            self.last_player = -1
            self.passes = 0
        return -1

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(90, 64)
        self.fc2 = nn.Linear(64, 15)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

def select_model_action(model, state, actions):
    """模型选择动作"""
    with torch.no_grad():
        probs = model(torch.FloatTensor(state))
    scores = [sum(probs[i].item() for i in range(15) if action[i] > 0) for action in actions]
    return int(np.argmax(scores)) if scores else 0

def select_rule_action(hand, actions, last_play):
    """规则Bot：小牌优先策略"""
    if len(actions) == 1:
        return 0
    
    # 优先出小牌
    valid_actions = [(i, a) for i, a in enumerate(actions) if a.sum() > 0]
    if valid_actions:
        # 找最小点数的牌
        min_idx = min(valid_actions, key=lambda x: np.where(x[1]>0)[0].min())[0]
        return min_idx
    return len(actions) - 1  # 过牌

def test(model_file='self_play_model.pt', num_games=100):
    """测试模型 vs 规则Bot"""
    print("=" * 50)
    print(f"测试: 模型 vs 规则Bot ({num_games}局)")
    print("=" * 50)
    
    # 加载模型
    model = PolicyNet()
    if model_file and np.os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
        print(f"加载模型: {model_file}")
    
    model_wins = 0
    rule_wins = 0
    
    for game_idx in range(num_games):
        game = FastGame()
        game.reset()
        
        while True:
            player = game.current
            actions = game.get_actions(player)
            state = game.get_state(player)
            
            # 红队(0,2,4)用模型，蓝队(1,3,5)用规则Bot
            if player % 2 == 0:
                action_idx = select_model_action(model, state, actions)
            else:
                action_idx = select_rule_action(game.hands[player], actions, game.last_play)
            
            action = actions[action_idx]
            result = game.step(player, action)
            
            if result >= 0:
                if result == 0:
                    model_wins += 1
                else:
                    rule_wins += 1
                break
            
            game.current = (game.current + 1) % 6
            while game.finished[game.current]:
                game.current = (game.current + 1) % 6
        
        if (game_idx + 1) % 20 == 0:
            print(f"[{game_idx+1}/{num_games}] 模型胜率: {model_wins/(game_idx+1)*100:.1f}%")
    
    print("=" * 50)
    print(f"最终结果:")
    print(f"  模型胜率: {model_wins/num_games*100:.1f}%")
    print(f"  规则Bot胜率: {rule_wins/num_games*100:.1f}%")
    print("=" * 50)
    
    return model_wins / num_games * 100

if __name__ == '__main__':
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'self_play_model.pt'
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    test(model_file, num_games)

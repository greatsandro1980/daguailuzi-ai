"""
V14b 策略优化版推理服务
基于V13架构，优化三带二、配合度、大小王策略
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============== V14b ActorCritic模型 (与V13相同架构) ==============
class ActorCritic(nn.Module):
    """V14b Actor-Critic网络 (与V13相同架构)"""
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


# ============== 简化游戏环境 ==============
class SimpleGame:
    """简化版游戏环境用于推理"""
    def __init__(self):
        self.hands = np.zeros((6, 15), dtype=np.int32)
        self.current = 0
        self.last_play = None
        self.last_player = -1
        self.passes = 0
        self.finished = [False] * 6
        
    def get_state(self, hand, current_played, hand_sizes, seat_index, trump_rank):
        """生成状态向量"""
        s = np.zeros(60, dtype=np.float32)
        
        # 编码手牌 (0-14)
        for card in hand:
            rank = self._get_card_rank(card, trump_rank)
            if 0 <= rank < 15:
                s[rank] += 1
        
        # 编码上家出的牌 (15-29)
        if current_played:
            for card in current_played:
                rank = self._get_card_rank(card, trump_rank)
                if 0 <= rank < 15:
                    s[15 + rank] += 1
        
        # 编码各玩家手牌数 (30-35)
        for i, size in enumerate(hand_sizes):
            s[30 + i] = size / 27.0
        
        # 其他特征
        if current_played:
            s[39] = 0.0  # 不是首发
        else:
            s[39] = 1.0  # 首发
        
        # 手牌统计
        single_count = sum(1 for i in range(13) if s[i] == 1)
        pair_count = sum(1 for i in range(13) if s[i] == 2)
        triple_count = sum(1 for i in range(13) if s[i] >= 3)
        s[46] = single_count
        s[47] = pair_count
        s[48] = triple_count
        s[49] = s[13]  # 小王
        s[50] = s[14]  # 大王
        s[51] = sum(s[:15])  # 总牌数
        
        return s
    
    def _get_card_rank(self, card, trump_rank):
        """获取牌的等级 (0-14)"""
        if isinstance(card, dict):
            if card.get('isBigJoker', False) or card.get('is_big_joker', False):
                return 14
            if card.get('isSmallJoker', False) or card.get('is_small_joker', False):
                return 13
            rank = card.get('rank', 2)
            # 2-14 映射到 0-12
            return max(0, min(12, rank - 2))
        return 0


def get_legal_actions(hand, current_played, trump_rank, is_first_play):
    """获取合法动作列表"""
    actions = []
    
    # 统计手牌
    card_counts = {}
    for card in hand:
        if isinstance(card, dict):
            if card.get('isBigJoker', False) or card.get('is_big_joker', False):
                rank = 14
            elif card.get('isSmallJoker', False) or card.get('is_small_joker', False):
                rank = 13
            else:
                rank = card.get('rank', 2)
        else:
            rank = 2
        card_counts[rank] = card_counts.get(rank, 0) + 1
    
    if is_first_play or not current_played:
        # 首发：可以出任意单张、对子、三张
        for rank, count in card_counts.items():
            if rank <= 14:
                actions.append((rank, 1))  # 单张
            if count >= 2 and rank <= 12:
                actions.append((rank, 2))  # 对子
            if count >= 3 and rank <= 12:
                actions.append((rank, 3))  # 三张
    else:
        # 压牌：需要出比上家大的同类型牌
        last_rank, last_count = _get_last_play_info(current_played)
        
        for rank, count in card_counts.items():
            if rank > last_rank and count >= last_count:
                actions.append((rank, last_count))
        
        # 可以pass
        actions.append((-1, 0))
    
    return actions if actions else [(-1, 0)]


def _get_last_play_info(current_played):
    """获取上家出牌信息"""
    if not current_played:
        return -1, 1
    
    ranks = []
    for card in current_played:
        if isinstance(card, dict):
            if card.get('isBigJoker', False) or card.get('is_big_joker', False):
                ranks.append(14)
            elif card.get('isSmallJoker', False) or card.get('is_small_joker', False):
                ranks.append(13)
            else:
                ranks.append(card.get('rank', 2))
    
    if not ranks:
        return -1, 1
    
    return max(ranks), len(ranks)


# ============== 全局变量 ==============
net = None
device = None
game = SimpleGame()


def load_model(ckpt_path):
    global net, device
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if ckpt_path and os.path.exists(ckpt_path):
        net = ActorCritic(hidden_dim=512).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        print(f"✅ V14b模型已加载: {ckpt_path}")
    else:
        print(f"⚠️ 模型文件不存在: {ckpt_path}")
        net = ActorCritic(hidden_dim=512).to(device)
        net.eval()


def select_action(hand, current_played, hand_sizes, seat_index, trump_rank, is_first_play):
    """使用模型选择动作"""
    global net, device, game
    
    # 获取合法动作
    actions = get_legal_actions(hand, current_played, trump_rank, is_first_play)
    
    if not actions:
        return [], True
    
    # 生成状态
    state = game.get_state(hand, current_played, hand_sizes, seat_index, trump_rank)
    state_t = torch.FloatTensor(state).to(device)
    
    # 模型推理
    with torch.no_grad():
        logits, _ = net(state_t)
    
    # 获取有效动作的logits
    valid = [15 if c < 0 else c for c, _ in actions]
    probs = torch.softmax(logits, 0)
    valid_probs = torch.tensor([max(probs[idx].item(), 1e-8) for idx in valid])
    valid_probs = valid_probs / valid_probs.sum()
    
    # 选择最佳动作
    best_idx = valid_probs.argmax().item()
    card_rank, count = actions[best_idx]
    
    if card_rank < 0:
        return [], True  # pass
    
    # 转换回牌对象
    action_cards = []
    remaining = count
    
    for card in hand:
        if remaining <= 0:
            break
        if isinstance(card, dict):
            if card.get('isBigJoker', False) or card.get('is_big_joker', False):
                r = 14
            elif card.get('isSmallJoker', False) or card.get('is_small_joker', False):
                r = 13
            else:
                r = card.get('rank', 2)
            
            if r == card_rank:
                action_cards.append(card)
                remaining -= 1
    
    return action_cards, len(action_cards) == 0


# ============== API路由 ==============
@app.route('/ai_action', methods=['POST'])
def ai_action():
    """AI动作接口"""
    data = request.get_json()
    
    seat_index = data['seatIndex']
    hand = data['hand']
    current_played = data.get('currentPlayed', [])
    hand_sizes = data.get('handSizes', [27] * 6)
    trump_rank = data.get('trumpRank', 2)
    is_first_play = data.get('isFirstPlay', False)
    
    action_cards, is_pass = select_action(
        hand, current_played, hand_sizes, seat_index, trump_rank, is_first_play
    )
    
    return jsonify({
        'action': action_cards,
        'pass': is_pass
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device), 'model': 'V14b_Strategy_Optimized'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/workspace/projects/rl_v14b_best.pt')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    
    load_model(args.model)
    print(f"V14b 策略优化版推理服务启动在 http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

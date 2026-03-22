"""
大怪路子 V9 AI 推理服务
使用简化的状态表示和神经网络进行决策
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 简化的神经网络架构
class DaguaiNetV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action_scores(self, state):
        """获取所有动作的分数"""
        with torch.no_grad():
            logits = self(torch.FloatTensor(state))
            return logits.numpy()

# 全局模型
model = None
device = None

def load_model(ckpt_path):
    global model, device
    device = torch.device('cpu')  # CPU推理足够快
    
    if ckpt_path and os.path.exists(ckpt_path):
        model = DaguaiNetV9()
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        print(f"✅ V9模型已加载: {ckpt_path}")
    else:
        print(f"⚠️ 模型文件不存在: {ckpt_path}")
        model = None

def card_to_rank(card):
    """从牌对象获取点数 (2-14, 小王=13, 大王=14)"""
    if isinstance(card, dict):
        if card.get('is_small_joker') or card.get('isSmallJoker'):
            return 13
        if card.get('is_big_joker') or card.get('isBigJoker'):
            return 14
        return card.get('rank', 2)
    return 2

def card_to_id(card):
    """获取牌的唯一标识"""
    if isinstance(card, dict):
        return card.get('id', 0)
    return 0

def encode_state_v9(hand, current_played, hand_sizes, seat_index, trump_rank, is_first_play):
    """
    编码游戏状态为60维向量
    - hand: 手牌列表
    - current_played: 当前需要压的牌
    - hand_sizes: 各玩家手牌数
    - seat_index: 当前玩家位置
    - trump_rank: 主牌点数
    - is_first_play: 是否首发
    """
    state = np.zeros(60, dtype=np.float32)
    
    # 1. 手牌统计 (0-14): 每种点数的数量
    hand_counts = np.zeros(15, dtype=np.float32)
    for card in hand:
        rank = card_to_rank(card)
        if rank >= 2 and rank <= 14:
            hand_counts[rank - 2] += 1
    state[:15] = hand_counts / 4.0  # 归一化
    
    # 2. 当前出牌 (15-29): 需要压的牌
    if current_played and len(current_played) > 0:
        played_counts = np.zeros(15, dtype=np.float32)
        for card in current_played:
            rank = card_to_rank(card)
            if rank >= 2 and rank <= 14:
                played_counts[rank - 2] += 1
        state[15:30] = played_counts / 4.0
    
    # 3. 各玩家手牌数 (30-35)
    for i, size in enumerate(hand_sizes[:6]):
        state[30 + i] = size / 27.0
    
    # 4. 当前出牌类型 (36-38): 单张/对子/三张
    if current_played and len(current_played) > 0:
        cnt = len(current_played)
        if cnt == 1:
            state[36] = 1.0
        elif cnt == 2:
            state[37] = 1.0
        elif cnt >= 3:
            state[38] = 1.0
    
    # 5. 是否首发 (39)
    state[39] = 1.0 if is_first_play else 0.0
    
    # 6. 队友状态 (40-42): 队友是否已出完
    team = seat_index % 2
    teammate_idx = 0
    for p in range(6):
        if p % 2 == team and p != seat_index:
            if hand_sizes[p] == 0:
                state[40 + teammate_idx] = 1.0
            teammate_idx += 1
    
    # 7. 对手状态 (43-45): 对手是否已出完
    opponent_idx = 0
    for p in range(6):
        if p % 2 != team:
            if hand_sizes[p] == 0:
                state[43 + opponent_idx] = 1.0
            opponent_idx += 1
    
    # 8. 手牌特征 (46-51)
    # 单张数量
    state[46] = sum(1 for c in hand_counts if c == 1) / 13.0
    # 对子数量
    state[47] = sum(1 for c in hand_counts if c == 2) / 13.0
    # 三张数量
    state[48] = sum(1 for c in hand_counts if c >= 3) / 13.0
    # 小王数量
    state[49] = hand_counts[11] / 2.0  # rank 13 = index 11
    # 大王数量
    state[50] = hand_counts[12] / 2.0  # rank 14 = index 12
    # 总牌数
    state[51] = len(hand) / 27.0
    
    # 剩余填充0
    return state

def get_legal_actions(hand, current_played, trump_rank, is_first_play):
    """
    获取合法动作列表
    返回: [(点数, 数量), ...] 或 [(-1, 0)] 表示pass
    """
    hand_counts = {}
    for card in hand:
        rank = card_to_rank(card)
        if rank >= 2 and rank <= 14:
            hand_counts[rank] = hand_counts.get(rank, 0) + 1
    
    actions = []
    
    if is_first_play or not current_played or len(current_played) == 0:
        # 首发: 可以出单张、对子、三张
        for rank in range(2, 15):  # 2到14
            count = hand_counts.get(rank, 0)
            if count >= 1:
                actions.append((rank, 1))  # 单张
            if count >= 2:
                actions.append((rank, 2))  # 对子
            if count >= 3:
                actions.append((rank, 3))  # 三张
    else:
        # 压牌: 需要出更大的同类型牌
        played_rank = max(card_to_rank(c) for c in current_played)
        played_count = len(current_played)
        
        for rank in range(played_rank + 1, 15):  # 比played_rank大
            count = hand_counts.get(rank, 0)
            if count >= played_count:
                actions.append((rank, played_count))
        
        # 可以pass
        actions.append((-1, 0))
    
    return actions if actions else [(-1, 0)]

def action_to_cards(hand, action, current_played):
    """
    将动作转换为具体的牌
    action: (点数, 数量) 或 (-1, 0) 表示pass
    """
    if action[0] == -1:
        return []  # pass
    
    target_rank = action[0]
    target_count = action[1]
    
    # 找出符合点数的牌
    result = []
    for card in hand:
        if len(result) >= target_count:
            break
        rank = card_to_rank(card)
        if rank == target_rank:
            result.append(card)
    
    return result

@app.route('/ai_action', methods=['POST'])
def ai_action():
    """
    AI推理接口
    请求体:
    {
        "seatIndex": 2,
        "hand": [...],             // 当前AI的手牌
        "currentPlayed": [...],    // 当前桌面需要压的牌
        "playedAll": [...],        // 已打出的所有牌
        "handSizes": [27,27,...],  // 每位玩家剩余张数
        "trumpRank": 2,            // 当前主牌点数
        "isFirstPlay": false
    }
    响应:
    {
        "action": [...],           // 出的牌
        "pass": false              // 是否pass
    }
    """
    try:
        data = request.get_json()
        
        seat_index = data.get('seatIndex', 0)
        hand = data.get('hand', [])
        current_played = data.get('currentPlayed', [])
        hand_sizes = data.get('handSizes', [27]*6)
        trump_rank = data.get('trumpRank', 2)
        is_first_play = data.get('isFirstPlay', False)
        
        print(f"[V9 AI] 座位{seat_index}, 手牌{len(hand)}张, 首发{is_first_play}")
        
        # 获取合法动作
        actions = get_legal_actions(hand, current_played, trump_rank, is_first_play)
        print(f"[V9 AI] 合法动作: {actions}")
        
        if not actions:
            return jsonify({'action': [], 'pass': True})
        
        # 如果只有pass选项
        if len(actions) == 1 and actions[0][0] == -1:
            return jsonify({'action': [], 'pass': True})
        
        # 使用模型选择动作
        if model is not None:
            state = encode_state_v9(hand, current_played, hand_sizes, seat_index, trump_rank, is_first_play)
            scores = model.get_action_scores(state)
            
            # 计算每个合法动作的分数
            action_scores = []
            valid_actions = [a for a in actions if a[0] != -1]  # 非pass动作
            
            for act in actions:
                if act[0] == -1:
                    # pass动作：只有在没有其他选择时才选择
                    if valid_actions:
                        score = -1000  # 有其他选择时，大幅惩罚pass
                    else:
                        score = (scores[15] if len(scores) > 15 else 0)
                else:
                    # 使用对应点数的输出
                    idx = act[0] - 2  # rank 2-14 -> index 0-12
                    if 0 <= idx < 15:
                        score = scores[idx]
                        # 鼓励出小牌（保留大牌）
                        # 但如果能压住对手，给予奖励
                        if not is_first_play and current_played:
                            played_rank = max(card_to_rank(c) for c in current_played)
                            if act[0] > played_rank:
                                score += 5.0  # 奖励能压住的牌
                    else:
                        score = -100
                action_scores.append((act, score))
            
            # 选择分数最高的动作
            best_action = max(action_scores, key=lambda x: x[1])[0]
            print(f"[V9 AI] 模型选择: {best_action}, 分数: {max(s for _, s in action_scores):.3f}")
        else:
            # 无模型，使用简单策略
            # 优先出小牌
            valid_actions = [a for a in actions if a[0] != -1]
            if valid_actions:
                best_action = min(valid_actions, key=lambda x: x[0])
            else:
                best_action = (-1, 0)
            print(f"[V9 AI] 规则选择: {best_action}")
        
        # 转换为具体牌
        selected_cards = action_to_cards(hand, best_action, current_played)
        
        return jsonify({
            'action': selected_cards,
            'pass': len(selected_cards) == 0
        })
        
    except Exception as e:
        print(f"[V9 AI] 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'action': [], 'pass': True, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'V9',
        'device': str(device)
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default=os.path.join(os.path.dirname(__file__), '..', 'rl_v9_best_rule.pt'))
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    
    load_model(args.model)
    print(f"V9 AI推理服务启动在 http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

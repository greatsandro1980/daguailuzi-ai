"""
大怪路子 AI 推理服务
启动一个 HTTP 服务，游戏服务器调用它来获取 AI 决策
"""
import os
import json
import argparse
import torch
import numpy as np
from flask import Flask, request, jsonify

from game_env import (
    encode_state, legal_actions, cards_to_vec,
    RANK_SMALL_JOKER, RANK_BIG_JOKER, TEAM_MAP
)
from model import DaguaiNet
from model import select_action as ac_select_action
from model_ppo import DaguaiPPONet
from model_ppo import select_action as ppo_select_action
from card_play_module_v2_optimized import FastCardPlayModule
import torch.nn.functional as F

app = Flask(__name__)

# 全局网络（懒加载）
net         = None
device      = None
is_ppo_mode = False   # True=PPO网络, False=旧AC网络


def select_action_auto(net, obs, device, greedy=True):
    """自动根据模型类型调用对应的 select_action"""
    if is_ppo_mode:
        action, _, _, _, _ = ppo_select_action(net, obs, device, greedy=greedy)
    else:
        action, _, _ = ac_select_action(net, obs, device, greedy=greedy)
    return action


is_stage2_mode = False  # True=stage2 FastCardPlayModule

def load_model(ckpt_path):
    global net, device, is_ppo_mode, is_stage2_mode
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # 判断模型类型
        if 'stage2' in os.path.basename(ckpt_path).lower() or 'self_play' in os.path.basename(ckpt_path).lower():
            # 使用FastCardPlayModule
            is_stage2_mode = True
            is_ppo_mode = False
            net = FastCardPlayModule(hidden_dim=256, action_dim=512).to(device)
            ep = ckpt.get('episode', 0)
            print(f"✅ Stage2模型已加载: {ckpt_path}  (ep {ep})")
            state_dict = ckpt.get('model_state_dict', ckpt)
            net.load_state_dict(state_dict)
        elif 'ppo' in os.path.basename(ckpt_path).lower():
            is_ppo_mode = True
            is_stage2_mode = False
            net = DaguaiPPONet(hidden_dim=512).to(device)
            ep = ckpt.get('episode', 0)
            print(f"✅ PPO模型已加载: {ckpt_path}  (ep {ep})")
            state_dict = ckpt.get('model', ckpt)
            net.load_state_dict(state_dict)
        else:
            is_ppo_mode = False
            is_stage2_mode = False
            net = DaguaiNet().to(device)
            ep = ckpt.get('episode', 0)
            print(f"✅ AC模型已加载: {ckpt_path}  (ep {ep})")
            state_dict = ckpt.get('model', ckpt)
            net.load_state_dict(state_dict)
    else:
        print("⚠️  未找到模型文件，使用随机权重（AC网络）")
        is_ppo_mode = False
        is_stage2_mode = False
        net = DaguaiNet().to(device)
    net.eval()


_JS_CARD_RE = __import__('re').compile(r'deck(\d+)-card-(\d+)')

def js_card_to_py(js_card):
    card_id = js_card.get('id', '')
    if isinstance(card_id, int):
        return card_id
    m = _JS_CARD_RE.match(str(card_id))
    if m:
        return (int(m.group(1)) - 1) * 54 + int(m.group(2))
    return 0


# Stage2模型编码函数
def encode_hand_stage2(hand):
    """将手牌编码为tensor [27]"""
    indices = []
    for c in hand[:27]:
        if isinstance(c, int):
            indices.append(c)
        elif isinstance(c, dict):
            indices.append(c.get('id', 0))
        else:
            indices.append(0)
    while len(indices) < 27:
        indices.append(0)
    return torch.tensor(indices[:27], dtype=torch.long)

def encode_actions_stage2(actions):
    """将动作列表编码为tensor [20, 512]"""
    candidates = []
    for action in actions[:20]:
        feat = torch.zeros(512)
        if action:
            for i, card in enumerate(action[:5]):
                if isinstance(card, int):
                    feat[i * 100 + card % 100] = 1.0
                elif isinstance(card, dict):
                    feat[i * 100 + card.get('id', 0) % 100] = 1.0
        candidates.append(feat)
    while len(candidates) < 20:
        candidates.append(torch.zeros(512))
    return torch.stack(candidates)

def encode_game_state_stage2(hand_sizes, seat_index, trump_rank):
    """编码游戏状态"""
    features = []
    # 玩家位置 one-hot
    player_feat = [0] * 6
    player_feat[seat_index] = 1
    features.extend(player_feat)
    # 已出完人数
    finished = sum(1 for h in hand_sizes if h == 0)
    features.append(finished / 6.0)
    # 各玩家手牌数
    features.extend([h / 27.0 for h in hand_sizes])
    # 主牌等级
    features.append(trump_rank / 14.0)
    return torch.tensor(features, dtype=torch.float32)

@app.route('/ai_action', methods=['POST'])
def ai_action():
    """
    请求体（JSON）：
    {
      "seatIndex": 2,
      "hand": [...],             // 当前 AI 的手牌（JS Card 格式）
      "currentPlayed": [...],    // 当前桌面需要压的牌（空数组=首发）
      "playedAll": [...],        // 已打出的所有牌
      "handSizes": [27,27,...],  // 每位玩家剩余张数
      "trumpRank": 2,            // 当前主牌点数
      "isFirstPlay": false
    }
    响应：
    {
      "action": [...],           // 出的牌（JS Card 格式）
      "pass": false              // 是否 pass
    }
    """
    data = request.get_json()

    seat_index   = data['seatIndex']
    hand_js      = data['hand']
    current_js   = data.get('currentPlayed', [])
    played_js    = data.get('playedAll', [])
    hand_sizes   = data.get('handSizes', [27]*6)
    trump_rank   = data.get('trumpRank', 2)
    is_first_play= data.get('isFirstPlay', False)

    # 转换牌格式
    hand         = [js_card_to_py(c) for c in hand_js]
    current_played = [js_card_to_py(c) for c in current_js]
    played_all   = [js_card_to_py(c) for c in played_js]

    # Stage2模型推理
    if is_stage2_mode:
        legal = legal_actions(hand, current_played, trump_rank, is_first_play)
        if not legal:
            return jsonify({'action': [], 'pass': True})
        
        with torch.no_grad():
            hand_t = encode_hand_stage2(hand).unsqueeze(0).to(device)
            actions_t = encode_actions_stage2(legal).unsqueeze(0).to(device)
            state_t = encode_game_state_stage2(hand_sizes, seat_index, trump_rank).unsqueeze(0).to(device)
            values = net(hand_t, actions_t, state_t)
            values = values.squeeze(0)
            valid_values = values[:len(legal)]
            action_idx = valid_values.argmax().item()
        
        action = legal[action_idx]
        
        # 转换回JS格式
        if not action:
            return jsonify({'action': [], 'pass': True})
        
        py_id_to_js = {js_card_to_py(c): c for c in hand_js}
        action_js = [py_id_to_js.get(aid) for aid in action if aid in py_id_to_js]
        action_js = [c for c in action_js if c is not None]
        
        return jsonify({
            'action': action_js,
            'pass':   len(action_js) == 0,
        })
    
    # 原有的DaguaiNet/PPO推理
    obs = {
        'current_player': seat_index,
        'hand':           hand,
        'played_all':     played_all,
        'current_played': current_played,
        'hand_sizes':     hand_sizes,
        'trump_rank':     trump_rank,
        'my_team':        TEAM_MAP[seat_index],
        'legal_actions':  legal_actions(hand, current_played, trump_rank, is_first_play),
    }

    # 用网络选择动作（greedy 模式，不探索）
    action = select_action_auto(net, obs, device, greedy=True)

    # action 是 Python 整数 id 列表，转回 JS 格式（通过反向映射匹配原始牌对象）
    # 建立 py_id -> js_card 映射
    py_id_to_js = {js_card_to_py(c): c for c in hand_js}
    action_js = [py_id_to_js[aid] for aid in action if aid in py_id_to_js]

    return jsonify({
        'action': action_js,
        'pass':   len(action_js) == 0,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'checkpoints_ppo/ppo_latest.pt'))
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()

    load_model(args.model)
    print(f"推理服务启动在 http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)

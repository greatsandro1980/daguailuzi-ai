"""
优化的游戏环境 - 快速合法动作生成
"""

import random
import numpy as np
from itertools import combinations
from copy import deepcopy
from game_env import (
    RANKS, SUITS, RANK_SMALL_JOKER, RANK_BIG_JOKER, TEAM_MAP,
    CardType, TYPE_ORDER, make_deck, rank_value, recognize, can_beat,
    cards_to_vec, encode_state, FEATURE_DIM
)


def fast_legal_actions(hand, current_played, trump_rank=None, is_first_play=False, max_actions=100):
    """
    快速生成合法动作 - 限制数量避免爆炸
    """
    if is_first_play or not current_played:
        return _fast_all_valid_combos(hand, trump_rank, max_actions)
    else:
        n = len(current_played)
        actions = _fast_combos_of_size(hand, n, trump_rank, max_actions)
        # 过滤能压过的
        beat_actions = [c for c in actions if can_beat(current_played, c, trump_rank)]
        beat_actions.append([])  # pass
        return beat_actions


def _fast_combos_of_size(hand, n, trump_rank, max_actions=30):
    """快速生成n张牌的合法组合 - 严格限制迭代次数"""
    if len(hand) < n:
        return []
    
    scored_combos = []
    iteration_count = 0
    max_iterations = {1: 27, 2: 100, 3: 200, 5: 500}
    
    for combo in combinations(hand, n):
        iteration_count += 1
        if iteration_count > max_iterations.get(n, 100):
            break
        
        combo = list(combo)
        if recognize(combo, trump_rank):
            score = _score_combo(combo, trump_rank)
            scored_combos.append((combo, score))
            if len(scored_combos) >= max_actions:
                break
    
    # 按分数排序，保留高分动作
    scored_combos.sort(key=lambda x: x[1], reverse=True)
    return [combo for combo, _ in scored_combos[:max_actions]]


def _score_combo(combo, trump_rank):
    """给牌型打分，用于优先保留高价值动作"""
    if not combo:
        return 0
    
    score = 0
    n = len(combo)
    
    # 基础分：牌数量（鼓励出完牌）
    score += n * 0.1
    
    # 识别牌型
    rec = recognize(combo, trump_rank)
    if rec:
        card_type, main_rank = rec
        # 大牌型加分
        type_scores = {
            'five_of_kind': 5, 'straight_flush': 4, 'four_with_one': 3,
            'three_with_two': 2.5, 'flush': 1.5, 'straight': 1,
            'triple': 0.8, 'pair': 0.5, 'single': 0.2
        }
        score += type_scores.get(card_type, 0)
    
    # 小牌优先（平均rank越小越好）
    avg_rank = sum(c.get('rank', 0) for c in combo) / len(combo)
    score += max(0, (15 - avg_rank)) * 0.1
    
    return score


def _fast_all_valid_combos(hand, trump_rank, max_actions=50):
    """快速生成所有合法牌型组合 - 严格限制迭代次数"""
    all_combos = []
    
    # 严格限制每种类型的最大迭代次数
    max_iterations = {1: 27, 2: 100, 3: 200, 5: 500}  # 1/2/3/5张牌的最大迭代次数
    max_per_type = max_actions // 4  # 每种类型最多保留数量
    
    for n in [1, 2, 3, 5]:
        if len(hand) < n:
            continue
        
        type_combos = []
        iteration_count = 0
        
        for combo in combinations(hand, n):
            iteration_count += 1
            if iteration_count > max_iterations.get(n, 100):
                break
            
            combo = list(combo)
            if recognize(combo, trump_rank):
                score = _score_combo(combo, trump_rank)
                type_combos.append((combo, score))
                if len(type_combos) >= max_per_type:
                    break
        
        # 按分数排序，保留高分动作
        type_combos.sort(key=lambda x: x[1], reverse=True)
        for combo, _ in type_combos[:max_per_type]:
            all_combos.append(combo)
    
    return all_combos


class FastDaguaiEnv:
    """优化的游戏环境"""
    
    def __init__(self, max_actions_per_step=50):
        self.max_actions = max_actions_per_step
        
    def reset(self):
        deck = make_deck()
        random.shuffle(deck)
        self.hands = [deck[i*27:(i+1)*27] for i in range(6)]
        self.trump_rank = 2
        self.current_player = 0
        self.current_played = []
        self.round_starter = 0
        self.finish_order = []
        self.played_all = []
        self.pass_count = 0
        self.is_first_play = True
        self.done = False
        self.scores = [0, 0]
        # 新增：记录对手上一步出牌（用于奖励计算）
        self.last_enemy_action = None
        self.last_enemy_seat = None
        return self._get_obs()
    
    def _get_obs(self):
        cp = self.current_player
        actions = fast_legal_actions(
            self.hands[cp],
            self.current_played,
            self.trump_rank,
            self.is_first_play,
            self.max_actions
        )
        return {
            'current_player': cp,
            'hand': self.hands[cp],
            'played_all': self.played_all,
            'current_played': self.current_played,
            'hand_sizes': [len(h) for h in self.hands],
            'trump_rank': self.trump_rank,
            'my_team': TEAM_MAP[cp],
            'legal_actions': actions,
            'finish_order': list(self.finish_order),
            # 新增：用于奖励计算
            'last_enemy_action': self.last_enemy_action,
            'last_enemy_seat': self.last_enemy_seat,
        }
    
    def step(self, action):
        cp = self.current_player
        rewards = [0.0] * 6
        
        # 记录对手上一步出牌（用于奖励计算）
        if cp in [1, 3, 5]:  # B队是对手
            self.last_enemy_seat = cp
            self.last_enemy_action = action
        
        if action:
            action_ids = {c['id'] for c in action}
            self.hands[cp] = [c for c in self.hands[cp] if c['id'] not in action_ids]
            self.played_all.extend(action)
            self.current_played = action
            self.round_starter = cp
            self.pass_count = 0
            self.is_first_play = False
            
            if len(self.hands[cp]) == 0:
                self.finish_order.append(cp)
                rewards[cp] += 0.3
        else:
            self.pass_count += 1
        
        active = 6 - len(self.finish_order)
        if self.pass_count >= active - 1 and self.current_played:
            self.current_played = []
            self.pass_count = 0
            self.is_first_play = True
            nxt = self._next_active(self.round_starter)
            self.current_player = nxt
        else:
            self.current_player = self._next_active(cp)
        
        self.done = len(self.finish_order) >= 5
        if self.done:
            rewards = self._calc_final_rewards()
        
        obs = self._get_obs()
        return obs, rewards, self.done, {}
    
    def _next_active(self, from_seat):
        finished = set(self.finish_order)
        for i in range(1, 7):
            nxt = (from_seat + i) % 6
            if nxt not in finished:
                return nxt
        return from_seat
    
    def _calc_final_rewards(self):
        rewards = [0.0] * 6
        r_map = [1.0, 0.3, 0.0, -0.3, -0.6, -1.0]
        
        all_order = list(self.finish_order)
        remaining = [i for i in range(6) if i not in all_order]
        all_order.extend(remaining)
        
        for rank, seat in enumerate(all_order):
            rewards[seat] = r_map[rank]
        
        winner_team = TEAM_MAP[all_order[0]]
        loser_team = TEAM_MAP[all_order[-1]]
        for seat in range(6):
            if TEAM_MAP[seat] == winner_team:
                rewards[seat] += 0.5
            if TEAM_MAP[seat] == loser_team:
                rewards[seat] -= 0.5
        
        return rewards


if __name__ == '__main__':
    import time
    env = FastDaguaiEnv()
    obs = env.reset()
    print(f"初始化成功，合法动作数: {len(obs['legal_actions'])}")
    
    steps = 0
    start = time.time()
    while not env.done:
        actions = obs['legal_actions']
        action = random.choice(actions)
        obs, rewards, done, _ = env.step(action)
        steps += 1
    
    elapsed = time.time() - start
    print(f"游戏结束，{steps}步，用时{elapsed:.2f}秒")

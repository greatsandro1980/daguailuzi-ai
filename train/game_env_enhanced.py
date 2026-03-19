"""
大怪路子增强版游戏环境
改进：特征增强 + 奖励塑形 + 对手建模
"""
import random
import numpy as np
from itertools import combinations
from collections import defaultdict
from copy import deepcopy

# 复用原始常量
from game_env import (
    RANKS, SUITS, RANK_SMALL_JOKER, RANK_BIG_JOKER,
    TEAM_MAP, CardType, TYPE_ORDER, make_deck,
    recognize, can_beat, rank_value
)

# ─── 增强版特征维度 ─────────────────────────────────
# 基础: 54*4 = 216
# 牌权: 6*6 = 36 (6轮 * 6玩家)
# 对手牌型统计: 6*9 = 54 (每个玩家每种牌型出过几次)
# 队友状态: 6 (队友手牌数、是否被压制)
# 剩余牌分布: 54 (还有哪些牌在外面)
# 总计: 216 + 36 + 54 + 6 + 54 = 366
FEATURE_DIM_ENHANCED = 366


# ─── 增强版编码 ─────────────────────────────────────
def cards_to_vec(cards):
    """牌组转54维向量"""
    from game_env import cards_to_vec as base_vec
    return base_vec(cards)


def encode_state_enhanced(obs, game_history=None):
    """
    增强版状态编码
    obs: 原始观测
    game_history: 游戏历史记录（用于计算牌权等）
    """
    # 1. 基础特征 (216)
    from game_env import encode_state
    base_feat = encode_state(obs)
    
    # 2. 牌权信息 (36): 6轮 * 6玩家
    control_feat = np.zeros(36, dtype=np.float32)
    if game_history and 'control_history' in game_history:
        for i, controller in enumerate(game_history['control_history'][:6]):
            if 0 <= controller < 6:
                control_feat[i * 6 + controller] = 1.0
    
    # 3. 对手牌型统计 (54)
    opponent_stats = np.zeros(54, dtype=np.float32)
    if game_history and 'card_type_history' in game_history:
        for seat, type_counts in game_history['card_type_history'].items():
            type_list = [
                CardType.SINGLE, CardType.PAIR, CardType.TRIPLE,
                CardType.STRAIGHT, CardType.FLUSH, CardType.THREE_WITH_TWO,
                CardType.FOUR_WITH_ONE, CardType.STRAIGHT_FLUSH, CardType.FIVE_OF_KIND
            ]
            for j, ct in enumerate(type_list):
                count = type_counts.get(ct, 0)
                opponent_stats[seat * 9 + j] = min(count / 5.0, 1.0)  # 归一化
    
    # 4. 队友状态 (6)
    teammate_feat = np.zeros(6, dtype=np.float32)
    my_team = obs['my_team']
    hand_sizes = obs['hand_sizes']
    
    # 找队友
    teammates = [s for s in range(6) if TEAM_MAP[s] == my_team and s != obs['current_player']]
    if teammates:
        t = teammates[0]
        teammate_feat[0] = hand_sizes[t] / 27.0  # 队友手牌比例
        teammate_feat[1] = 1.0 if hand_sizes[t] <= 5 else 0.0  # 队友是否快出完
        teammate_feat[2] = 1.0 if hand_sizes[t] <= 3 else 0.0  # 队友是否危险
        # 队友是否被压制（上轮pass）
        if game_history and 'last_pass' in game_history:
            teammate_feat[3] = 1.0 if game_history['last_pass'] == t else 0.0
    
    # 5. 剩余牌分布 (54)
    remaining_feat = np.zeros(54, dtype=np.float32)
    hand_vec = cards_to_vec(obs['hand'])
    played_vec = cards_to_vec(obs['played_all'])
    # 剩余 = 1 - 手牌 - 已出牌 (粗略估计，实际每张牌有3张)
    remaining_feat = np.clip(1.0 - hand_vec - played_vec / 3.0, 0, 1)
    
    return np.concatenate([
        base_feat, control_feat, opponent_stats, teammate_feat, remaining_feat
    ])


# ─── 奖励塑形 ───────────────────────────────────────
def compute_step_reward(env, action, prev_obs, game_history=None):
    """
    中间步骤奖励（关键改进！）
    """
    reward = 0.0
    cp = env.current_player
    
    # 1. pass 小惩罚
    if not action:
        return -0.02
    
    # 2. 出牌效率奖励
    cards_played = len(action)
    reward += 0.01 * cards_played
    
    # 3. 牌权控制奖励（重要！）
    ct = recognize(action, env.trump_rank)
    if ct:
        type_order = TYPE_ORDER.get(ct[0], 0)
        # 大牌型获得牌权的奖励更多
        base_control_reward = 0.03 + 0.01 * type_order
        
        # 如果这手牌能压过上一手，奖励更多
        if env.current_played and can_beat(env.current_played, action, env.trump_rank):
            reward += base_control_reward * 1.5
        else:
            reward += base_control_reward
    
    # 4. 保护队友奖励（关键策略！）
    my_team = TEAM_MAP[cp]
    teammates = [s for s in range(6) if TEAM_MAP[s] == my_team and s != cp]
    
    for t in teammates:
        teammate_hand = env.hands[t]
        if len(teammate_hand) <= 5 and len(teammate_hand) > 0:
            # 队友牌少，这手牌可能帮ta脱困
            reward += 0.05
    
    # 5. 压制对手奖励
    opponents = [s for s in range(6) if TEAM_MAP[s] != my_team]
    for o in opponents:
        opp_hand = env.hands[o]
        if len(opp_hand) <= 5 and len(opp_hand) > 0:
            # 压制快出完的对手
            reward += 0.03
    
    # 6. 留牌策略奖励（保留大牌）
    if len(env.hands[cp]) > 0:
        remaining = env.hands[cp]
        # 评估剩余牌的质量
        quality = evaluate_hand_quality(remaining, env.trump_rank)
        if quality > 0.6:
            reward += 0.02  # 保留了好牌
    
    # 7. 出完牌奖励
    if len(env.hands[cp]) == 0:
        reward += 0.5  # 大奖励
    
    return reward


def evaluate_hand_quality(hand, trump_rank=None):
    """
    评估手牌质量（0-1分）
    考虑：大牌、王、好牌型
    """
    if not hand:
        return 0.0
    
    score = 0.0
    
    # 大王/小王加分
    big_jokers = sum(1 for c in hand if c.get('is_big_joker'))
    small_jokers = sum(1 for c in hand if c.get('is_small_joker'))
    score += big_jokers * 0.15 + small_jokers * 0.1
    
    # 大牌加分 (A, K, 将牌)
    for c in hand:
        if c.get('rank') == 14:  # A
            score += 0.02
        elif c.get('rank') == 13:  # K
            score += 0.01
        if trump_rank and c.get('rank') == trump_rank:
            score += 0.03
    
    # 对子/三条加分
    rank_counts = defaultdict(int)
    for c in hand:
        if not c.get('is_joker'):
            rank_counts[c['rank']] += 1
    
    for r, cnt in rank_counts.items():
        if cnt >= 4:
            score += 0.1
        elif cnt == 3:
            score += 0.05
        elif cnt == 2:
            score += 0.02
    
    return min(score, 1.0)


# ─── 增强版游戏环境 ─────────────────────────────────
class DaguaiEnvEnhanced:
    """增强版环境，记录更多历史信息"""
    
    def __init__(self):
        self.reset()
    
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
        
        # 增强版历史记录
        self.game_history = {
            'control_history': [],      # 谁控制了牌权
            'card_type_history': {i: defaultdict(int) for i in range(6)},  # 每人出过的牌型
            'last_pass': -1,            # 上一个pass的玩家
            'round_plays': [],          # 本轮出牌记录
        }
        
        return self._get_obs()
    
    def _get_obs(self):
        cp = self.current_player
        actions = self._legal_actions(
            self.hands[cp],
            self.current_played,
            self.trump_rank,
            self.is_first_play
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
            'game_history': self.game_history,
        }
    
    def _legal_actions(self, hand, current_played, trump_rank, is_first_play):
        """生成合法动作"""
        from game_env import legal_actions
        return legal_actions(hand, current_played, trump_rank, is_first_play)
    
    def step(self, action):
        """执行一步，返回 (obs, reward, done, info)"""
        cp = self.current_player
        prev_obs = self._get_obs()
        
        # 计算中间奖励
        step_reward = compute_step_reward(self, action, prev_obs, self.game_history)
        
        if action:
            # 出牌
            action_ids = {c['id'] for c in action}
            self.hands[cp] = [c for c in self.hands[cp] if c['id'] not in action_ids]
            self.played_all.extend(action)
            self.current_played = action
            self.round_starter = cp
            self.pass_count = 0
            self.is_first_play = False
            
            # 记录牌型
            ct = recognize(action, self.trump_rank)
            if ct:
                self.game_history['card_type_history'][cp][ct[0]] += 1
            
            # 记录牌权
            self.game_history['control_history'].append(cp)
            if len(self.game_history['control_history']) > 6:
                self.game_history['control_history'].pop(0)
            
            # 记录本轮出牌
            self.game_history['round_plays'].append((cp, action))
            
            # 检查出完
            if len(self.hands[cp]) == 0:
                self.finish_order.append(cp)
        
        else:
            # pass
            self.pass_count += 1
            self.game_history['last_pass'] = cp
        
        # 判断本轮结束
        active = 6 - len(self.finish_order)
        if self.pass_count >= active - 1 and self.current_played:
            self.current_played = []
            self.pass_count = 0
            self.is_first_play = True
            self.game_history['round_plays'] = []
            nxt = self._next_active(self.round_starter)
            self.current_player = nxt
        else:
            self.current_player = self._next_active(cp)
        
        # 判断游戏结束
        self.done = len(self.finish_order) >= 5
        final_rewards = [0.0] * 6
        
        if self.done:
            final_rewards = self._calc_final_rewards()
            # 中间奖励 + 最终奖励
            rewards = [step_reward + final_rewards[i] if i == cp else final_rewards[i] 
                       for i in range(6)]
        else:
            rewards = [step_reward if i == cp else 0.0 for i in range(6)]
        
        obs = self._get_obs()
        return obs, rewards, self.done, {'step_reward': step_reward}
    
    def _next_active(self, from_seat):
        finished = set(self.finish_order)
        for i in range(1, 7):
            nxt = (from_seat + i) % 6
            if nxt not in finished:
                return nxt
        return from_seat
    
    def _calc_final_rewards(self):
        """计算最终奖励（增强版）"""
        rewards = [0.0] * 6
        r_map = [1.5, 0.8, 0.3, -0.3, -0.8, -1.5]  # 增大差距
        
        all_order = list(self.finish_order)
        remaining = [i for i in range(6) if i not in all_order]
        all_order.extend(remaining)
        
        for rank, seat in enumerate(all_order):
            rewards[seat] = r_map[rank]
        
        # 队伍胜负奖励
        winner_team = TEAM_MAP[all_order[0]]
        loser_team = TEAM_MAP[all_order[-1]]
        for seat in range(6):
            if TEAM_MAP[seat] == winner_team:
                rewards[seat] += 0.5
            if TEAM_MAP[seat] == loser_team:
                rewards[seat] -= 0.3
        
        return rewards
    
    def get_feature_enhanced(self):
        """获取增强版特征"""
        return encode_state_enhanced(self._get_obs(), self.game_history)


# ─── 测试 ───────────────────────────────────────────
if __name__ == '__main__':
    env = DaguaiEnvEnhanced()
    obs = env.reset()
    print(f"增强版特征维度: {encode_state_enhanced(obs, env.game_history).shape}")
    
    # 跑一局
    steps = 0
    total_rewards = [0.0] * 6
    while not env.done:
        actions = obs['legal_actions']
        action = random.choice(actions)
        obs, rewards, done, info = env.step(action)
        for i in range(6):
            total_rewards[i] += rewards[i]
        steps += 1
    
    print(f"\n游戏结束，共 {steps} 步")
    print(f"完成顺序: {env.finish_order}")
    print(f"累计奖励: {[f'{r:.2f}' for r in total_rewards]}")

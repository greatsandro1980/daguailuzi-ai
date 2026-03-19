"""
大怪路子游戏环境
完整实现发牌、出牌、进贡、胜负判断逻辑
"""
import random
import numpy as np
from itertools import combinations
from copy import deepcopy

# ─── 牌面常量 ─────────────────────────────────────────
RANKS = list(range(2, 15))        # 2~14 (A=14)
SUITS = ['♠', '♥', '♣', '♦']
RANK_SMALL_JOKER = 16
RANK_BIG_JOKER   = 17
TEAM_MAP = [0, 1, 0, 1, 0, 1]    # seat 0,2,4 → 红队(0); 1,3,5 → 蓝队(1)

# 牌型
class CardType:
    SINGLE        = 'single'
    PAIR          = 'pair'
    TRIPLE        = 'triple'
    STRAIGHT      = 'straight'
    FLUSH         = 'flush'
    THREE_WITH_TWO= 'three_with_two'
    FOUR_WITH_ONE = 'four_with_one'
    STRAIGHT_FLUSH= 'straight_flush'
    FIVE_OF_KIND  = 'five_of_kind'

TYPE_ORDER = {
    CardType.SINGLE:         1,
    CardType.PAIR:           2,
    CardType.TRIPLE:         3,
    CardType.STRAIGHT:       4,
    CardType.FLUSH:          5,
    CardType.THREE_WITH_TWO: 6,
    CardType.FOUR_WITH_ONE:  7,
    CardType.STRAIGHT_FLUSH: 8,
    CardType.FIVE_OF_KIND:   9,
}


# ─── 一张牌：用整数编码 0~161 ─────────────────────────
# 3副牌：deck 0/1/2，每副 54 张 (52 普通 + 小王 + 大王)
# card_id = deck * 54 + suit_idx * 13 + rank_idx   (普通牌)
# card_id = deck * 54 + 52                          (小王)
# card_id = deck * 54 + 53                          (大王)

def make_deck():
    """生成3副牌，每张牌是 dict"""
    cards = []
    for d in range(3):
        for si, suit in enumerate(SUITS):
            for ri, rank in enumerate(RANKS):
                cards.append({
                    'id': d * 54 + si * 13 + ri,
                    'deck': d,
                    'suit': suit,
                    'rank': rank,
                    'is_joker': False,
                    'is_big_joker': False,
                    'is_small_joker': False,
                })
        # 小王
        cards.append({
            'id': d * 54 + 52,
            'deck': d,
            'suit': 'joker-black',
            'rank': RANK_SMALL_JOKER,
            'is_joker': True,
            'is_big_joker': False,
            'is_small_joker': True,
        })
        # 大王
        cards.append({
            'id': d * 54 + 53,
            'deck': d,
            'suit': 'joker-red',
            'rank': RANK_BIG_JOKER,
            'is_joker': True,
            'is_big_joker': True,
            'is_small_joker': False,
        })
    return cards


def rank_value(rank, trump_rank=None):
    if rank == RANK_BIG_JOKER:   return 100
    if rank == RANK_SMALL_JOKER: return 99
    if trump_rank and rank == trump_rank: return 98
    return rank


# ─── 牌型识别 ─────────────────────────────────────────
def recognize(cards, trump_rank=None):
    """返回 (CardType, main_rank) 或 None"""
    n = len(cards)
    if n == 0: return None

    jokers    = [c for c in cards if c['is_joker']]
    non_jokers= [c for c in cards if not c['is_joker']]
    jc        = len(jokers)

    def rank_counts(lst):
        d = {}
        for c in lst:
            d[c['rank']] = d.get(c['rank'], 0) + 1
        return d

    if n == 1:
        return (CardType.SINGLE, cards[0]['rank'])

    if n == 2:
        rc = rank_counts(non_jokers)
        # 两张相同 or 1张+1王
        candidates = []
        for r, cnt in rc.items():
            if cnt + jc >= 2:
                candidates.append(r)
        if jc == 2:
            candidates.append(RANK_BIG_JOKER)
        if candidates:
            return (CardType.PAIR, min(candidates))
        return None

    if n == 3:
        rc = rank_counts(non_jokers)
        candidates = []
        for r, cnt in rc.items():
            if cnt + jc >= 3:
                candidates.append(r)
        if jc == 3:
            candidates.append(RANK_BIG_JOKER)
        if candidates:
            return (CardType.TRIPLE, min(candidates))
        return None

    if n == 5:
        rc = rank_counts(non_jokers)
        suits_used = list(set(c['suit'] for c in non_jokers))

        # 五同
        for r, cnt in rc.items():
            if cnt + jc >= 5:
                return (CardType.FIVE_OF_KIND, r)
        if jc >= 5:
            return (CardType.FIVE_OF_KIND, RANK_BIG_JOKER)

        # 同花顺
        if len(suits_used) <= 1:
            nj_ranks = sorted(c['rank'] for c in non_jokers)
            for start in range(2, 11):
                needed = sum(1 for r in range(start, start+5) if r not in nj_ranks)
                if needed <= jc:
                    return (CardType.STRAIGHT_FLUSH, start + 4)

        # 四带一
        for r, cnt in rc.items():
            needed = 4 - cnt
            if jc >= needed:
                rem_jc = jc - needed
                others = [c for c in non_jokers if c['rank'] != r]
                if len(others) + rem_jc >= 1:
                    return (CardType.FOUR_WITH_ONE, r)

        # 三带二
        for r, cnt in rc.items():
            needed3 = 3 - cnt
            if jc >= needed3:
                rem_jc = jc - needed3
                others = {k: v for k, v in rc.items() if k != r}
                for r2, cnt2 in others.items():
                    if cnt2 + rem_jc >= 2:
                        return (CardType.THREE_WITH_TWO, r)
                if rem_jc >= 2:
                    return (CardType.THREE_WITH_TWO, r)

        # 同花
        if len(suits_used) <= 1 and len(non_jokers) + jc == 5:
            return (CardType.FLUSH, max(c['rank'] for c in non_jokers) if non_jokers else RANK_BIG_JOKER)

        # 顺子
        nj_ranks = sorted(c['rank'] for c in non_jokers)
        for start in range(2, 11):
            needed = sum(1 for r in range(start, start+5) if r not in nj_ranks)
            if needed <= jc:
                return (CardType.STRAIGHT, start + 4)

    return None


def can_beat(played, challenger, trump_rank=None):
    """challenger 能否压过 played，返回 bool"""
    pt = recognize(played, trump_rank)
    ct = recognize(challenger, trump_rank)
    if pt is None or ct is None: return False
    if len(played) != len(challenger): return False
    po, co = TYPE_ORDER[pt[0]], TYPE_ORDER[ct[0]]
    if co != po: return co > po
    return rank_value(ct[1], trump_rank) > rank_value(pt[1], trump_rank)


# ─── 合法动作生成 ─────────────────────────────────────
def legal_actions(hand, current_played, trump_rank=None, is_first_play=False):
    """
    返回所有合法出牌动作列表，每个动作是一个 card list。
    空列表 [] 代表"不出/pass"。
    """
    if is_first_play or not current_played:
        # 首发：枚举所有合法牌型（1/2/3/5张）
        return _all_valid_combos(hand, trump_rank)
    else:
        n = len(current_played)
        # 必须出同数量且能压过的牌，或者 pass
        actions = [c for c in _combos_of_size(hand, n, trump_rank)
                   if can_beat(current_played, c, trump_rank)]
        actions.append([])   # 可以选择不出
        return actions


def _combos_of_size(hand, n, trump_rank):
    result = []
    for combo in combinations(hand, n):
        combo = list(combo)
        if recognize(combo, trump_rank):
            result.append(combo)
    return result


def _all_valid_combos(hand, trump_rank):
    result = []
    for n in [1, 2, 3, 5]:
        if len(hand) < n: continue
        for combo in combinations(hand, n):
            combo = list(combo)
            if recognize(combo, trump_rank):
                result.append(combo)
    return result


# ─── 特征编码（神经网络输入） ────────────────────────────
# 总维度 = 54*4 + 6 + 2 = 224
#   54*4: 自己手牌 / 已出牌（所有人）/ 当前桌面牌 / 合法动作掩码
#   6: 每位玩家剩余张数（归一化）
#   2: 主牌rank归一化 / 我的队伍
FEATURE_DIM = 54 * 4 + 6 + 2   # = 224
CARD_VEC_IDX = 0  # 占位符

def cards_to_vec(cards):
    """把一组牌转成 54 维 0/1 向量（按deck+rank去重后取OR）"""
    vec = np.zeros(54, dtype=np.float32)
    for c in cards:
        if c['is_big_joker']:
            vec[53] = 1.0
        elif c['is_small_joker']:
            vec[52] = 1.0
        else:
            si = SUITS.index(c['suit'])
            ri = RANKS.index(c['rank'])
            vec[si * 13 + ri] = 1.0
    return vec


def encode_state(obs):
    """
    obs: dict
      - hand: list of cards
      - played_all: list of cards (所有人已打出的牌，扁平)
      - current_played: list of cards (当前桌面需要压的牌)
      - hand_sizes: list[6] (每位玩家手牌数)
      - trump_rank: int or None
      - my_team: int (0/1)
    返回 shape=(FEATURE_DIM,) 的 numpy 数组
    """
    hand_vec     = cards_to_vec(obs['hand'])
    played_vec   = cards_to_vec(obs['played_all'])
    current_vec  = cards_to_vec(obs['current_played'])
    # 合法动作并集（告诉网络有哪些牌可打）
    legal_cards = []
    for action in obs.get('legal_actions', []):
        legal_cards.extend(action)
    legal_vec = cards_to_vec(legal_cards)

    hand_sizes = np.array(obs['hand_sizes'], dtype=np.float32) / 27.0  # 归一化
    trump_feat = np.array([obs['trump_rank'] / 17.0 if obs['trump_rank'] else 0.0], dtype=np.float32)
    team_feat  = np.array([float(obs['my_team'])], dtype=np.float32)

    return np.concatenate([hand_vec, played_vec, current_vec, legal_vec,
                           hand_sizes, trump_feat, team_feat])


# ─── 游戏主环境 ───────────────────────────────────────
class DaguaiEnv:
    """
    6人大怪路子环境。
    每次 step 只推进一个玩家的决策。
    """

    def reset(self):
        deck = make_deck()
        random.shuffle(deck)
        # 每人 27 张（3副162张 ÷ 6）
        self.hands = [deck[i*27:(i+1)*27] for i in range(6)]
        self.trump_rank = 2          # 初始主牌是 2，可扩展为升级逻辑
        self.current_player = 0
        self.current_played = []     # 当前轮需要压的牌
        self.round_starter  = 0      # 本轮首发玩家
        self.finish_order   = []     # 出完牌的顺序
        self.played_all     = []     # 所有已打出的牌（用于特征）
        self.pass_count     = 0      # 连续 pass 计数
        self.is_first_play  = True
        self.done           = False
        self.scores         = [0, 0] # [红队, 蓝队]
        return self._get_obs()

    def _get_obs(self):
        cp = self.current_player
        actions = legal_actions(
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
        }

    def step(self, action):
        """
        action: list of cards（出的牌），[] 表示 pass
        返回: (obs, reward[6], done, info)
        """
        cp = self.current_player
        rewards = [0.0] * 6

        if action:  # 出牌
            # 从手牌移除
            action_ids = {c['id'] for c in action}
            self.hands[cp] = [c for c in self.hands[cp] if c['id'] not in action_ids]
            self.played_all.extend(action)
            self.current_played = action
            self.round_starter   = cp
            self.pass_count      = 0
            self.is_first_play   = False

            # 检查是否出完
            if len(self.hands[cp]) == 0:
                self.finish_order.append(cp)
                rewards[cp] += 0.3   # 出完牌小奖励

        else:  # pass
            self.pass_count += 1

        # 判断本轮是否结束（其他人都 pass 了）
        active = 6 - len(self.finish_order)
        if self.pass_count >= active - 1 and self.current_played:
            # 本轮结束，round_starter 重新首发
            self.current_played = []
            self.pass_count = 0
            self.is_first_play  = True
            # 找下一个还有牌的玩家（从 round_starter 开始）
            nxt = self._next_active(self.round_starter)
            self.current_player = nxt
        else:
            self.current_player = self._next_active(cp)

        # 判断游戏结束
        self.done = len(self.finish_order) >= 5  # 5人出完，最后一人定局
        if self.done:
            rewards = self._calc_final_rewards()

        obs = self._get_obs()
        return obs, rewards, self.done, {}

    def _next_active(self, from_seat):
        """找下一个手里还有牌的玩家"""
        finished = set(self.finish_order)
        for i in range(1, 7):
            nxt = (from_seat + i) % 6
            if nxt not in finished:
                return nxt
        return from_seat

    def _calc_final_rewards(self):
        """
        根据完成顺序计算奖励。
        头游(+1.0) 二游(+0.3) 三游(0) 四游(-0.3) 五游(-0.6) 末游(-1.0)
        队伍胜负额外 ±0.5
        """
        rewards = [0.0] * 6
        r_map = [1.0, 0.3, 0.0, -0.3, -0.6, -1.0]

        # 确定末游（最后一个出完的人）
        all_order = list(self.finish_order)
        remaining = [i for i in range(6) if i not in all_order]
        all_order.extend(remaining)  # 末游

        for rank, seat in enumerate(all_order):
            rewards[seat] = r_map[rank]

        # 队伍胜负（头游队伍 +0.5 全队，末游队伍 -0.5 全队）
        winner_team = TEAM_MAP[all_order[0]]
        loser_team  = TEAM_MAP[all_order[-1]]
        for seat in range(6):
            if TEAM_MAP[seat] == winner_team:
                rewards[seat] += 0.5
            if TEAM_MAP[seat] == loser_team:
                rewards[seat] -= 0.5

        return rewards

    def get_feature(self):
        """返回当前玩家的特征向量"""
        return encode_state(self._get_obs())


# ─── 简单测试 ─────────────────────────────────────────
if __name__ == '__main__':
    env = DaguaiEnv()
    obs = env.reset()
    print(f"初始化成功，当前玩家: {obs['current_player']}")
    print(f"手牌张数: {obs['hand_sizes']}")
    print(f"合法动作数: {len(obs['legal_actions'])}")
    print(f"特征向量维度: {encode_state(obs).shape}")

    # 跑一局随机对弈
    steps = 0
    while not env.done:
        actions = obs['legal_actions']
        action  = random.choice(actions)
        obs, rewards, done, _ = env.step(action)
        steps += 1

    print(f"\n游戏结束，共 {steps} 步")
    print(f"完成顺序: {env.finish_order}")
    print(f"最终奖励: {[f'{r:.2f}' for r in rewards]}")

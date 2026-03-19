"""
大怪路子高速训练环境
优化：numpy数组表示、预计算、批量处理
目标速度：>1局/秒
"""
import numpy as np
import random
from numba import jit, prange
from typing import List, Tuple, Optional

# ─── 常量 ─────────────────────────────────────────────
RANKS = np.arange(2, 15)  # 2~14 (A=14)
RANK_SMALL_JOKER = 16
RANK_BIG_JOKER = 17
N_CARDS = 162  # 3副牌
N_HAND = 54    # 手牌向量长度

# 牌型编码
TYPE_SINGLE = 1
TYPE_PAIR = 2
TYPE_TRIPLE = 3
TYPE_STRAIGHT = 4
TYPE_FLUSH = 5
TYPE_THREE_TWO = 6
TYPE_FOUR_ONE = 7
TYPE_STRAIGHT_FLUSH = 8
TYPE_FIVE_KIND = 9

TYPE_ORDER = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


# ─── 快速牌型识别 ────────────────────────────────────
@jit(nopython=True, cache=True)
def fast_recognize(cards: np.ndarray, n: int, trump_rank: int) -> Tuple[int, int]:
    """
    快速牌型识别
    cards: 手牌向量 [54] (每个位置0-3表示该牌的数量)
    n: 要出的牌数
    返回: (牌型, 主点数) 或 (0, 0)
    """
    if n == 0:
        return (0, 0)
    
    # 统计点数
    rank_counts = np.zeros(15, dtype=np.int32)  # 2-14, 15=小王, 16=大王
    suit_counts = np.zeros(4, dtype=np.int32)
    joker_count = 0
    
    for i in range(54):
        if cards[i] > 0:
            if i < 52:
                rank = 2 + (i % 13)
                suit = i // 13
                rank_counts[rank - 2] += cards[i]
                suit_counts[suit] += cards[i]
            elif i == 52:  # 小王
                joker_count += cards[i]
            else:  # 大王
                joker_count += cards[i]
    
    if n == 1:
        # 单张
        for r in range(13):
            if rank_counts[r] >= 1:
                return (TYPE_SINGLE, r + 2)
        if joker_count >= 1:
            return (TYPE_SINGLE, 17)  # 大王
        
    elif n == 2:
        # 对子
        for r in range(13):
            if rank_counts[r] >= 2:
                return (TYPE_PAIR, r + 2)
        # 两张王
        if joker_count >= 2:
            return (TYPE_PAIR, 17)
            
    elif n == 3:
        # 三条
        for r in range(13):
            if rank_counts[r] >= 3:
                return (TYPE_TRIPLE, r + 2)
        # 2王+1牌 或 1王+2牌
        if joker_count >= 1:
            for r in range(13):
                if rank_counts[r] >= 2:
                    return (TYPE_TRIPLE, r + 2)
            if joker_count >= 2:
                for r in range(13):
                    if rank_counts[r] >= 1:
                        return (TYPE_TRIPLE, r + 2)
                        
    elif n == 5:
        # 五张牌型
        # 检查五同
        for r in range(13):
            if rank_counts[r] >= 5:
                return (TYPE_FIVE_KIND, r + 2)
        
        # 检查四带一
        for r in range(13):
            if rank_counts[r] >= 4:
                return (TYPE_FOUR_ONE, r + 2)
        
        # 检查三带二
        for r in range(13):
            if rank_counts[r] >= 3:
                for r2 in range(13):
                    if r2 != r and rank_counts[r2] >= 2:
                        return (TYPE_THREE_TWO, r + 2)
        
        # 检查同花
        for s in range(4):
            if suit_counts[s] >= 5:
                # 检查顺子
                consecutive = 0
                for r in range(13):
                    if rank_counts[r] >= 1:
                        consecutive += 1
                        if consecutive >= 5:
                            return (TYPE_STRAIGHT_FLUSH, r + 2)
                    else:
                        consecutive = 0
        
        # 检查顺子
        consecutive = 0
        for r in range(13):
            if rank_counts[r] >= 1:
                consecutive += 1
                if consecutive >= 5:
                    return (TYPE_STRAIGHT, r + 2)
            else:
                consecutive = 0
        
        # 检查同花（不需要顺子）
        for s in range(4):
            if suit_counts[s] >= 5:
                max_rank = 2
                for r in range(13):
                    if rank_counts[r] >= 1:
                        max_rank = r + 2
                return (TYPE_FLUSH, max_rank)
        
        # 王参与的牌型
        if joker_count >= 1:
            # 王可以补充任意牌
            # 简化：王+四张 = 四带一
            non_joker = sum(rank_counts)
            if non_joker + joker_count >= 5:
                for r in range(13):
                    if rank_counts[r] >= 3:
                        return (TYPE_THREE_TWO, r + 2)
                    if rank_counts[r] >= 4:
                        return (TYPE_FOUR_ONE, r + 2)
    
    return (0, 0)


@jit(nopython=True, cache=True)
def fast_can_beat(prev_type: int, prev_rank: int, cur_type: int, cur_rank: int, trump_rank: int) -> bool:
    """快速判断能否压过"""
    if cur_type == 0:
        return False
    if cur_type != prev_type:
        return cur_type > prev_type
    
    # 相同牌型比点数
    def rank_val(r):
        if r == 17: return 100  # 大王
        if r == 16: return 99   # 小王
        if r == trump_rank: return 98
        return r
    
    return rank_val(cur_rank) > rank_val(prev_rank)


# ─── 高速游戏环境 ────────────────────────────────────
class FastDaguaiEnv:
    """高速训练环境"""
    
    __slots__ = ['hands', 'played', 'current', 'hand_sizes', 'done', 
                 'trump_rank', 'current_player', 'finish_order', 'steps']
    
    def __init__(self):
        self.hands = np.zeros((6, 54), dtype=np.int8)  # 6个玩家，每人54维向量
        self.played = np.zeros(54, dtype=np.int8)      # 已出的牌
        self.current = np.zeros(54, dtype=np.int8)     # 当前桌面牌
        self.hand_sizes = np.zeros(6, dtype=np.int8)   # 每人手牌数
        self.done = False
        self.trump_rank = 3  # 默认主牌3
        self.current_player = 0
        self.finish_order = []
        self.steps = 0
    
    def reset(self) -> dict:
        """重置游戏"""
        self.hands.fill(0)
        self.played.fill(0)
        self.current.fill(0)
        self.hand_sizes.fill(27)  # 每人27张
        self.done = False
        self.finish_order = []
        self.steps = 0
        
        # 随机发牌
        deck = np.arange(54, dtype=np.int8)
        deck = np.tile(deck, 3)  # 3副牌
        np.random.shuffle(deck)
        
        for i in range(6):
            start = i * 27
            for j in range(27):
                card = deck[start + j]
                self.hands[i, card] += 1
        
        self.current_player = 0
        self.trump_rank = random.randint(2, 14)
        
        return self._get_obs()
    
    def _get_obs(self) -> dict:
        """获取观测"""
        cp = self.current_player
        legal = self._get_legal_actions()
        
        return {
            'hand': self.hands[cp].copy(),
            'played_all': self.played.copy(),
            'current_played': self.current.copy(),
            'hand_sizes': self.hand_sizes.copy(),
            'trump_rank': self.trump_rank,
            'my_team': cp % 2,
            'current_player': cp,
            'legal_actions': legal,
        }
    
    def _get_legal_actions(self) -> List:
        """获取合法动作（优化版）"""
        cp = self.current_player
        hand = self.hands[cp]
        
        if np.sum(self.current) == 0:
            # 首发：生成所有合法牌型
            return self._gen_first_actions(hand)
        else:
            # 跟牌：生成能压过的动作
            return self._gen_beat_actions(hand)
    
    def _gen_first_actions(self, hand: np.ndarray) -> List:
        """首发动作生成"""
        actions = []
        
        # 单张
        for i in range(54):
            if hand[i] > 0:
                action = np.zeros(54, dtype=np.int8)
                action[i] = 1
                actions.append(action)
        
        # 对子
        for i in range(54):
            if hand[i] >= 2:
                action = np.zeros(54, dtype=np.int8)
                action[i] = 2
                actions.append(action)
        
        # 三条
        for i in range(54):
            if hand[i] >= 3:
                action = np.zeros(54, dtype=np.int8)
                action[i] = 3
                actions.append(action)
        
        # 五张牌型（简化：只生成常见牌型）
        # 三带二
        for i in range(13):
            if hand[i] >= 3:
                for j in range(13):
                    if j != i and hand[j] >= 2:
                        action = np.zeros(54, dtype=np.int8)
                        action[i] = 3
                        action[j] = 2
                        actions.append(action)
        
        # 四带一
        for i in range(13):
            if hand[i] >= 4:
                for j in range(54):
                    if j != i and hand[j] >= 1:
                        action = np.zeros(54, dtype=np.int8)
                        action[i] = 4
                        action[j] = 1
                        actions.append(action)
        
        # 顺子（简化：只生成几个常见顺子）
        for start in range(2, 10):  # 2-6 到 9-K
            ranks_needed = np.arange(start, start + 5)
            action = np.zeros(54, dtype=np.int8)
            valid = True
            for r in ranks_needed:
                if r < 15:
                    idx = r - 2  # 转换为索引
                    # 检查任意花色
                    found = False
                    for s in range(4):
                        card_idx = s * 13 + idx
                        if hand[card_idx] > 0:
                            action[card_idx] = 1
                            found = True
                            break
                    if not found:
                        valid = False
                        break
            if valid and np.sum(action) == 5:
                actions.append(action)
        
        return actions
    
    def _gen_beat_actions(self, hand: np.ndarray) -> List:
        """跟牌动作生成"""
        actions = []
        n = int(np.sum(self.current))
        
        if n == 1:
            # 找更大的单张
            prev_type, prev_rank = fast_recognize(self.current, 1, self.trump_rank)
            for i in range(54):
                if hand[i] > 0:
                    test = np.zeros(54, dtype=np.int8)
                    test[i] = 1
                    cur_type, cur_rank = fast_recognize(test, 1, self.trump_rank)
                    if fast_can_beat(prev_type, prev_rank, cur_type, cur_rank, self.trump_rank):
                        actions.append(test)
        
        elif n == 2:
            # 找更大的对子
            prev_type, prev_rank = fast_recognize(self.current, 2, self.trump_rank)
            for i in range(54):
                if hand[i] >= 2:
                    test = np.zeros(54, dtype=np.int8)
                    test[i] = 2
                    cur_type, cur_rank = fast_recognize(test, 2, self.trump_rank)
                    if fast_can_beat(prev_type, prev_rank, cur_type, cur_rank, self.trump_rank):
                        actions.append(test)
        
        elif n == 3:
            # 找更大的三条
            prev_type, prev_rank = fast_recognize(self.current, 3, self.trump_rank)
            for i in range(54):
                if hand[i] >= 3:
                    test = np.zeros(54, dtype=np.int8)
                    test[i] = 3
                    cur_type, cur_rank = fast_recognize(test, 3, self.trump_rank)
                    if fast_can_beat(prev_type, prev_rank, cur_type, cur_rank, self.trump_rank):
                        actions.append(test)
        
        elif n == 5:
            # 五张牌型
            prev_type, prev_rank = fast_recognize(self.current, 5, self.trump_rank)
            
            # 三带二
            for i in range(13):
                if hand[i] >= 3:
                    for j in range(13):
                        if j != i and hand[j] >= 2:
                            test = np.zeros(54, dtype=np.int8)
                            test[i] = 3
                            test[j] = 2
                            cur_type, cur_rank = fast_recognize(test, 5, self.trump_rank)
                            if fast_can_beat(prev_type, prev_rank, cur_type, cur_rank, self.trump_rank):
                                actions.append(test)
        
        # 可以pass
        actions.append(np.zeros(54, dtype=np.int8))
        
        return actions
    
    def step(self, action: np.ndarray) -> Tuple[dict, np.ndarray, bool, dict]:
        """执行动作"""
        cp = self.current_player
        
        # 出牌
        if np.sum(action) > 0:
            self.hands[cp] -= action
            self.played += action
            self.current = action.copy()
            self.hand_sizes[cp] = int(np.sum(self.hands[cp]))
        else:
            # pass
            pass
        
        # 检查是否出完
        if self.hand_sizes[cp] == 0:
            self.finish_order.append(cp)
            if len(self.finish_order) >= 5:
                self.done = True
        
        # 下一个玩家
        self.current_player = (self.current_player + 1) % 6
        while self.hand_sizes[self.current_player] == 0 and not self.done:
            self.current_player = (self.current_player + 1) % 6
        
        self.steps += 1
        if self.steps > 500:
            self.done = True
        
        # 计算奖励
        rewards = np.zeros(6, dtype=np.float32)
        if self.done:
            rewards = self._compute_final_rewards()
        
        return self._get_obs(), rewards, self.done, {}
    
    def _compute_final_rewards(self) -> np.ndarray:
        """计算终局奖励"""
        rewards = np.zeros(6, dtype=np.float32)
        
        # 根据完成顺序给奖励
        for i, player in enumerate(self.finish_order):
            rewards[player] = [1.5, 1.0, 0.5, -0.3, -0.7, -1.2][i]
        
        # 未完成的玩家
        for i in range(6):
            if i not in self.finish_order:
                rewards[i] = -1.0
        
        # 团队奖励
        red_team = rewards[0] + rewards[2] + rewards[4]
        blue_team = rewards[1] + rewards[3] + rewards[5]
        
        if red_team > blue_team:
            rewards[0] += 0.5
            rewards[2] += 0.5
            rewards[4] += 0.5
        else:
            rewards[1] += 0.5
            rewards[3] += 0.5
            rewards[5] += 0.5
        
        return rewards


# ─── 快速状态编码 ────────────────────────────────────
@jit(nopython=True, cache=True)
def fast_encode(hand: np.ndarray, played: np.ndarray, current: np.ndarray,
                hand_sizes: np.ndarray, trump_rank: int, my_team: int) -> np.ndarray:
    """快速状态编码"""
    # 54*4 + 6 + 2 = 224
    feat = np.zeros(224, dtype=np.float32)
    
    # 手牌
    feat[0:54] = hand.astype(np.float32)
    # 已出牌
    feat[54:108] = played.astype(np.float32) / 3.0
    # 当前桌面
    feat[108:162] = current.astype(np.float32)
    # 合法动作掩码（简化为手牌）
    feat[162:216] = (hand > 0).astype(np.float32)
    
    # 手牌数
    feat[216:222] = hand_sizes.astype(np.float32) / 27.0
    # 主牌和队伍
    feat[222] = trump_rank / 14.0
    feat[223] = float(my_team)
    
    return feat


# ─── 测试 ────────────────────────────────────────────
if __name__ == '__main__':
    import time
    
    print("测试高速环境...")
    
    env = FastDaguaiEnv()
    
    # 预热JIT
    print("预热JIT...")
    for _ in range(3):
        obs = env.reset()
        done = False
        while not done:
            actions = obs['legal_actions']
            action = random.choice(actions)
            obs, rewards, done, _ = env.step(action)
    
    # 正式测试
    print("开始速度测试...")
    n_games = 100
    start = time.time()
    
    for _ in range(n_games):
        obs = env.reset()
        done = False
        while not done:
            actions = obs['legal_actions']
            action = random.choice(actions)
            obs, rewards, done, _ = env.step(action)
    
    elapsed = time.time() - start
    speed = n_games / elapsed
    
    print(f"速度: {speed:.2f} 局/秒")
    print(f"每局耗时: {elapsed/n_games*1000:.1f} 毫秒")

"""
大怪路子高速训练环境 v2
修复：动作生成逻辑、游戏流程
"""
import numpy as np
import random
from typing import List, Tuple, Optional

# ─── 常量 ─────────────────────────────────────────────
RANK_SMALL_JOKER = 16
RANK_BIG_JOKER = 17

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


def cards_to_ranks(cards: np.ndarray) -> Tuple[np.ndarray, int]:
    """将牌向量转换为点数统计和王数
    返回: (rank_counts[13], joker_count)
    """
    rank_counts = np.zeros(13, dtype=np.int8)
    joker_count = 0
    
    for i in range(52):
        if cards[i] > 0:
            rank = i % 13
            rank_counts[rank] += cards[i]
    
    joker_count = cards[52] + cards[53]
    return rank_counts, joker_count


def recognize_cards(cards: np.ndarray, trump_rank: int = 3) -> Tuple[int, int]:
    """快速牌型识别
    返回: (牌型, 主点数) 或 (0, 0)
    """
    n = int(np.sum(cards))
    if n == 0:
        return (0, 0)
    
    rank_counts, joker_count = cards_to_ranks(cards)
    
    # 单张
    if n == 1:
        for r in range(13):
            if rank_counts[r] >= 1:
                return (TYPE_SINGLE, r + 2)
        if joker_count >= 1:
            return (TYPE_SINGLE, 17)  # 大王优先
    
    # 对子
    elif n == 2:
        for r in range(13):
            if rank_counts[r] >= 2:
                return (TYPE_PAIR, r + 2)
        if joker_count >= 2:
            return (TYPE_PAIR, 17)
    
    # 三条
    elif n == 3:
        for r in range(13):
            if rank_counts[r] >= 3:
                return (TYPE_TRIPLE, r + 2)
        # 王组合
        for r in range(13):
            if rank_counts[r] >= 2 and joker_count >= 1:
                return (TYPE_TRIPLE, r + 2)
        if joker_count >= 3:
            return (TYPE_TRIPLE, 17)
    
    # 五张牌型
    elif n == 5:
        # 五同
        for r in range(13):
            if rank_counts[r] >= 5:
                return (TYPE_FIVE_KIND, r + 2)
        
        # 四带一
        for r in range(13):
            if rank_counts[r] >= 4:
                return (TYPE_FOUR_ONE, r + 2)
        
        # 三带二
        for r in range(13):
            if rank_counts[r] >= 3:
                for r2 in range(13):
                    if r2 != r and rank_counts[r2] >= 2:
                        return (TYPE_THREE_TWO, r + 2)
        
        # 王参与的三带二
        for r in range(13):
            if rank_counts[r] >= 3 and joker_count >= 2:
                return (TYPE_THREE_TWO, r + 2)
        
        # 同花顺（简化检查）
        for suit in range(4):
            count = 0
            for r in range(13):
                idx = suit * 13 + r
                if cards[idx] > 0:
                    count += 1
            if count >= 5:
                # 检查是否连续
                consecutive = 0
                for r in range(13):
                    idx = suit * 13 + r
                    if cards[idx] > 0:
                        consecutive += 1
                        if consecutive >= 5:
                            return (TYPE_STRAIGHT_FLUSH, r + 2)
                    else:
                        consecutive = 0
        
        # 顺子
        consecutive = 0
        for r in range(13):
            if rank_counts[r] >= 1:
                consecutive += 1
                if consecutive >= 5:
                    return (TYPE_STRAIGHT, r + 2)
            else:
                consecutive = 0
        
        # 同花
        for suit in range(4):
            count = sum(cards[suit * 13 + r] for r in range(13))
            if count >= 5:
                max_r = max(r for r in range(13) if cards[suit * 13 + r] > 0)
                return (TYPE_FLUSH, max_r + 2)
    
    return (0, 0)


def can_beat_cards(prev: np.ndarray, cur: np.ndarray, trump_rank: int = 3) -> bool:
    """判断cur能否压过prev"""
    pt, pr = recognize_cards(prev, trump_rank)
    ct, cr = recognize_cards(cur, trump_rank)
    
    if ct == 0:
        return False
    if pt == 0:
        return True
    
    # 不同牌型，比较牌型大小
    if ct != pt:
        return ct > pt
    
    # 相同牌型，比较点数
    def rank_val(r):
        if r == 17: return 100
        if r == 16: return 99
        if r == trump_rank: return 98
        return r
    
    return rank_val(cr) > rank_val(pr)


# ─── 高速游戏环境 ────────────────────────────────────
class FastDaguaiEnvV2:
    """高速训练环境 v2"""
    
    __slots__ = ['hands', 'played', 'current', 'hand_sizes', 'done', 
                 'trump_rank', 'current_player', 'finish_order', 'steps',
                 'pass_count', 'last_player']
    
    def __init__(self):
        self.hands = np.zeros((6, 54), dtype=np.int8)
        self.played = np.zeros(54, dtype=np.int8)
        self.current = np.zeros(54, dtype=np.int8)
        self.hand_sizes = np.zeros(6, dtype=np.int8)
        self.done = False
        self.trump_rank = 3
        self.current_player = 0
        self.finish_order = []
        self.steps = 0
        self.pass_count = 0  # 连续pass计数
        self.last_player = -1  # 上一个出牌的玩家
    
    def reset(self) -> dict:
        """重置游戏"""
        self.hands.fill(0)
        self.played.fill(0)
        self.current.fill(0)
        self.done = False
        self.finish_order = []
        self.steps = 0
        self.pass_count = 0
        self.last_player = -1
        
        # 随机发牌
        deck = list(range(54)) * 3
        random.shuffle(deck)
        
        for i in range(6):
            start = i * 27
            for j in range(27):
                card = deck[start + j]
                self.hands[i, card] += 1
        
        self.hand_sizes = np.array([27] * 6, dtype=np.int8)
        self.current_player = 0
        self.trump_rank = random.randint(2, 14)
        
        return self._get_obs()
    
    def _get_obs(self) -> dict:
        """获取观测"""
        return {
            'hand': self.hands[self.current_player].copy(),
            'played_all': self.played.copy(),
            'current_played': self.current.copy(),
            'hand_sizes': self.hand_sizes.copy(),
            'trump_rank': self.trump_rank,
            'my_team': self.current_player % 2,
            'current_player': self.current_player,
            'legal_actions': self._get_legal_actions(),
        }
    
    def _get_legal_actions(self) -> List[np.ndarray]:
        """获取合法动作"""
        cp = self.current_player
        hand = self.hands[cp]
        
        # 如果所有人都pass了，当前玩家可以首发
        if np.sum(self.current) == 0 or self.pass_count >= 5:
            self.current.fill(0)  # 清空桌面
            self.pass_count = 0
            return self._gen_first_actions(hand)
        else:
            return self._gen_beat_actions(hand)
    
    def _gen_first_actions(self, hand: np.ndarray) -> List[np.ndarray]:
        """首发动作生成"""
        actions = []
        
        # 单张
        for i in range(54):
            if hand[i] > 0:
                a = np.zeros(54, dtype=np.int8)
                a[i] = 1
                actions.append(a)
        
        # 对子
        for i in range(54):
            if hand[i] >= 2:
                a = np.zeros(54, dtype=np.int8)
                a[i] = 2
                actions.append(a)
        
        # 三条
        for i in range(54):
            if hand[i] >= 3:
                a = np.zeros(54, dtype=np.int8)
                a[i] = 3
                actions.append(a)
        
        # 三带二
        for r in range(13):
            # 三张相同点数
            three_cards = []
            for s in range(4):
                idx = s * 13 + r
                for _ in range(min(3, hand[idx])):
                    three_cards.append(idx)
            
            if len(three_cards) >= 3:
                # 找对子
                for r2 in range(13):
                    if r2 == r:
                        continue
                    pair_cards = []
                    for s in range(4):
                        idx = s * 13 + r2
                        for _ in range(min(2, hand[idx])):
                            pair_cards.append(idx)
                    
                    if len(pair_cards) >= 2:
                        a = np.zeros(54, dtype=np.int8)
                        for idx in three_cards[:3]:
                            a[idx] += 1
                        for idx in pair_cards[:2]:
                            a[idx] += 1
                        actions.append(a)
        
        # 四带一
        for r in range(13):
            four_cards = []
            for s in range(4):
                idx = s * 13 + r
                for _ in range(min(4, hand[idx])):
                    four_cards.append(idx)
            
            if len(four_cards) >= 4:
                # 找单张
                for i in range(54):
                    if hand[i] > 0 and i not in four_cards[:4]:
                        a = np.zeros(54, dtype=np.int8)
                        for idx in four_cards[:4]:
                            a[idx] += 1
                        a[i] = 1
                        actions.append(a)
                        break
        
        # 顺子
        for start in range(9):  # 2-6 到 10-A
            cards = []
            for r in range(start, start + 5):
                # 找任意花色
                for s in range(4):
                    idx = s * 13 + r
                    if hand[idx] > 0:
                        cards.append(idx)
                        break
            if len(cards) == 5:
                a = np.zeros(54, dtype=np.int8)
                for idx in cards:
                    a[idx] = 1
                actions.append(a)
        
        return actions if actions else [np.zeros(54, dtype=np.int8)]
    
    def _gen_beat_actions(self, hand: np.ndarray) -> List[np.ndarray]:
        """跟牌动作生成"""
        actions = []
        n = int(np.sum(self.current))
        
        if n == 1:
            # 找更大的单张
            pt, pr = recognize_cards(self.current, self.trump_rank)
            for i in range(54):
                if hand[i] > 0:
                    a = np.zeros(54, dtype=np.int8)
                    a[i] = 1
                    ct, cr = recognize_cards(a, self.trump_rank)
                    if can_beat_cards(self.current, a, self.trump_rank):
                        actions.append(a)
        
        elif n == 2:
            # 找更大的对子
            for i in range(54):
                if hand[i] >= 2:
                    a = np.zeros(54, dtype=np.int8)
                    a[i] = 2
                    if can_beat_cards(self.current, a, self.trump_rank):
                        actions.append(a)
        
        elif n == 3:
            # 找更大的三条
            for i in range(54):
                if hand[i] >= 3:
                    a = np.zeros(54, dtype=np.int8)
                    a[i] = 3
                    if can_beat_cards(self.current, a, self.trump_rank):
                        actions.append(a)
        
        elif n == 5:
            pt, pr = recognize_cards(self.current, self.trump_rank)
            
            # 三带二
            for r in range(13):
                three_cards = []
                for s in range(4):
                    idx = s * 13 + r
                    for _ in range(min(3, hand[idx])):
                        three_cards.append(idx)
                
                if len(three_cards) >= 3:
                    for r2 in range(13):
                        if r2 == r:
                            continue
                        pair_cards = []
                        for s in range(4):
                            idx = s * 13 + r2
                            for _ in range(min(2, hand[idx])):
                                pair_cards.append(idx)
                        
                        if len(pair_cards) >= 2:
                            a = np.zeros(54, dtype=np.int8)
                            for idx in three_cards[:3]:
                                a[idx] += 1
                            for idx in pair_cards[:2]:
                                a[idx] += 1
                            if can_beat_cards(self.current, a, self.trump_rank):
                                actions.append(a)
        
        # 可以pass
        actions.append(np.zeros(54, dtype=np.int8))
        
        return actions
    
    def step(self, action: np.ndarray) -> Tuple[dict, np.ndarray, bool, dict]:
        """执行动作"""
        cp = self.current_player
        n = int(np.sum(action))
        
        if n > 0:
            # 出牌
            self.hands[cp] -= action
            self.played += action
            self.current = action.copy()
            self.hand_sizes[cp] = int(np.sum(self.hands[cp]))
            self.pass_count = 0
            self.last_player = cp
            
            # 检查是否出完
            if self.hand_sizes[cp] == 0:
                self.finish_order.append(cp)
                if len(self.finish_order) >= 5:
                    self.done = True
        else:
            # pass
            self.pass_count += 1
            
            # 如果5个人都pass，清空桌面
            if self.pass_count >= 5:
                self.current.fill(0)
                self.pass_count = 0
        
        # 下一个玩家
        if not self.done:
            self.current_player = (self.current_player + 1) % 6
            # 跳过已出完的玩家
            attempts = 0
            while self.hand_sizes[self.current_player] == 0 and attempts < 6:
                self.current_player = (self.current_player + 1) % 6
                attempts += 1
                if attempts >= 6:
                    self.done = True
                    break
        
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
        
        # 根据完成顺序
        order_rewards = [1.5, 1.0, 0.5, -0.3, -0.7, -1.2]
        for i, player in enumerate(self.finish_order):
            if i < len(order_rewards):
                rewards[player] = order_rewards[i]
        
        # 未完成的玩家
        for i in range(6):
            if i not in self.finish_order:
                rewards[i] = -1.0
        
        # 团队奖励
        red_score = sum(rewards[s] for s in [0, 2, 4])
        blue_score = sum(rewards[s] for s in [1, 3, 5])
        
        if red_score > blue_score:
            for s in [0, 2, 4]:
                rewards[s] += 0.3
        else:
            for s in [1, 3, 5]:
                rewards[s] += 0.3
        
        return rewards


# ─── 测试 ────────────────────────────────────────────
if __name__ == '__main__':
    import time
    
    print("测试高速环境v2...")
    
    env = FastDaguaiEnvV2()
    
    # 正式测试
    n_games = 100
    start = time.time()
    finished = 0
    
    for _ in range(n_games):
        obs = env.reset()
        done = False
        while not done:
            actions = obs['legal_actions']
            # 优先出牌而非pass
            play_actions = [a for a in actions if np.sum(a) > 0]
            if play_actions:
                action = max(play_actions, key=lambda x: int(np.sum(x)))
            else:
                action = actions[0] if actions else np.zeros(54, dtype=np.int8)
            obs, rewards, done, _ = env.step(action)
        if len(env.finish_order) > 0:
            finished += 1
    
    elapsed = time.time() - start
    speed = n_games / elapsed
    
    print(f"速度: {speed:.2f} 局/秒")
    print(f"正常结束率: {finished}/{n_games}")
    print(f"平均每局耗时: {elapsed/n_games*1000:.1f} 毫秒")

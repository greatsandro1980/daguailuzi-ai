"""
大怪路子出牌模块 (Card Play Module)
借鉴斗地主DouZero思路，专注于"手数最少+牌力最大"的出牌策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from game_env import FULL_DECK, cards_to_vec, CARD_RANK, CARD_SUIT

# 牌型编码
CARD_TYPE_SINGLE = 0   # 单张
CARD_TYPE_PAIR = 1     # 对子
CARD_TYPE_TRIPLE = 2   # 三张
CARD_TYPE_FIVE = 3     # 5张牌型

# 5张牌型子类型
FIVE_TYPE_STRAIGHT = 0      # 杂顺（连续不同花）
FIVE_TYPE_FLUSH = 1         # 烂同花（同花不连续）
FIVE_TYPE_FULL_HOUSE = 2    # 葫芦
FIVE_TYPE_FOUR_ONE = 3      # 4带1
FIVE_TYPE_STRAIGHT_FLUSH = 4 # 同花顺
FIVE_TYPE_BOMB = 5          # 炸弹×5
FIVE_TYPE_KING = 6          # 五张王


class CardPlayEncoder(nn.Module):
    """手牌编码器：将27张手牌编码为特征向量"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 牌面特征嵌入 (点数2-17, 花色0-4)
        # rank范围: 2-14(普通牌), 16(小王), 17(大王)
        # suit范围: 0-3(四种花色), 4(王牌无花色)
        self.rank_embed = nn.Embedding(18, 32)  # 0-17
        self.suit_embed = nn.Embedding(5, 16)   # 0-4
        
        # 手牌整体编码
        self.card_encoder = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # 注意力聚合多张牌
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, hand_cards: List[int]) -> torch.Tensor:
        """
        hand_cards: 手牌索引列表 (0-53)
        return: (hidden_dim,) 手牌特征向量
        """
        if not hand_cards:
            return torch.zeros(self.hidden_dim)
        
        # 每张牌编码
        card_features = []
        for card_idx in hand_cards:
            rank = CARD_RANK[card_idx]  # 使用正确的rank (2-17)
            suit = CARD_SUIT[card_idx]  # 使用正确的suit
            
            rank_feat = self.rank_embed(torch.tensor(int(rank), dtype=torch.long))
            suit_feat = self.suit_embed(torch.tensor(int(suit), dtype=torch.long))
            card_feat = torch.cat([rank_feat, suit_feat], dim=0)
            card_features.append(card_feat)
        
        card_features = torch.stack(card_features).unsqueeze(0)  # (1, n_cards, 48)
        
        # 编码每张牌
        encoded = self.card_encoder(card_features)  # (1, n_cards, hidden_dim)
        
        # 注意力聚合
        attn_out, _ = self.attention(encoded, encoded, encoded)
        
        # 平均池化得到手牌整体特征
        hand_feat = attn_out.mean(dim=1).squeeze(0)  # (hidden_dim,)
        
        return hand_feat


class ActionEncoder(nn.Module):
    """出牌动作编码器"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 动作类型嵌入
        self.type_embed = nn.Embedding(4, 32)  # 单/对/三/五
        self.five_subtype_embed = nn.Embedding(7, 32)  # 5张牌型子类型
        
        # 动作特征编码
        self.action_encoder = nn.Sequential(
            nn.Linear(64 + 5, 128),  # 类型嵌入 + 牌力特征
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
    def encode_action(self, action_cards: List[int], hand_cards: List[int]) -> torch.Tensor:
        """
        编码一个出牌动作
        action_cards: 要出的牌
        hand_cards: 当前手牌（用于计算剩余手牌质量）
        return: (hidden_dim,) 动作特征
        """
        n_cards = len(action_cards)
        
        # 确定动作类型
        if n_cards == 1:
            action_type = CARD_TYPE_SINGLE
            five_subtype = 0
        elif n_cards == 2:
            action_type = CARD_TYPE_PAIR
            five_subtype = 0
        elif n_cards == 3:
            action_type = CARD_TYPE_TRIPLE
            five_subtype = 0
        else:  # n_cards == 5
            action_type = CARD_TYPE_FIVE
            five_subtype = self._classify_five_type(action_cards)
        
        # 类型嵌入
        type_feat = self.type_embed(torch.tensor(action_type, dtype=torch.long))
        subtype_feat = self.five_subtype_embed(torch.tensor(five_subtype, dtype=torch.long))
        
        # 牌力特征
        power_features = self._compute_power_features(action_cards, hand_cards)
        
        # 合并编码
        combined = torch.cat([type_feat, subtype_feat, power_features], dim=0).unsqueeze(0)
        action_feat = self.action_encoder(combined).squeeze(0)
        
        return action_feat
    
    def _classify_five_type(self, cards: List[int]) -> int:
        """分类5张牌型"""
        ranks = sorted([CARD_RANK[c] for c in cards])
        suits = [CARD_SUIT[c] for c in cards]
        
        # 检查五张王 (rank 16=小王, 17=大王)
        if all(r >= 16 for r in ranks):
            return FIVE_TYPE_KING
        
        # 检查炸弹×5
        if len(set(ranks)) == 1:
            return FIVE_TYPE_BOMB
        
        # 检查同花顺
        is_flush = len(set(suits)) == 1
        is_straight = ranks == list(range(ranks[0], ranks[0] + 5))
        if is_flush and is_straight:
            return FIVE_TYPE_STRAIGHT_FLUSH
        
        # 检查4带1
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        if sorted(rank_counts.values()) == [1, 4]:
            return FIVE_TYPE_FOUR_ONE
        
        # 检查葫芦
        if sorted(rank_counts.values()) == [2, 3]:
            return FIVE_TYPE_FULL_HOUSE
        
        # 检查烂同花
        if is_flush:
            return FIVE_TYPE_FLUSH
        
        # 检查杂顺
        if is_straight:
            return FIVE_TYPE_STRAIGHT
        
        # 默认烂同花（5张同花色的兜底）
        return FIVE_TYPE_FLUSH
    
    def _compute_power_features(self, action_cards: List[int], hand_cards: List[int]) -> torch.Tensor:
        """计算牌力特征"""
        n_cards = len(action_cards)
        
        # 当前出牌的最大rank (使用CARD_RANK)
        max_rank = max(CARD_RANK[c] for c in action_cards)
        
        # 剩余手牌数
        remaining = len(hand_cards) - n_cards
        
        # 是否含王 (rank 16=小王, 17=大王)
        has_king = any(CARD_RANK[c] >= 16 for c in action_cards)
        
        # 大王数量（用于王牌比较）
        big_king_count = sum(1 for c in action_cards if CARD_RANK[c] == 17)
        
        # 归一化特征
        features = torch.tensor([
            n_cards / 5.0,           # 出牌张数占比
            max_rank / 17.0,         # 最大rank归一化 (最大17)
            remaining / 27.0,        # 剩余手牌占比
            float(has_king),         # 是否含王
            big_king_count / 3.0     # 大王数量归一化
        ], dtype=torch.float32)
        
        return features


class CardPlayModule(nn.Module):
    """
    出牌模块主网络
    输入：当前手牌
    输出：最优出牌动作
    """
    
    def __init__(self, hidden_dim=256, temperature=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        self.hand_encoder = CardPlayEncoder(hidden_dim)
        self.action_encoder = ActionEncoder(hidden_dim)
        
        # 动作评分网络
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, hand_cards: List[int]) -> Tuple[List[int], torch.Tensor]:
        """
        前向传播：选择最优出牌
        return: (选中的牌, 动作分数)
        """
        # 编码手牌
        hand_feat = self.hand_encoder(hand_cards)
        
        # 生成所有可能的出牌组合
        candidates = self._generate_candidates(hand_cards)
        
        if not candidates:
            return [], torch.tensor(0.0)
        
        # 评分每个候选动作
        scores = []
        for action in candidates:
            action_feat = self.action_encoder.encode_action(action, hand_cards)
            combined = torch.cat([hand_feat, action_feat], dim=0)
            score = self.scorer(combined)
            scores.append(score)
        
        scores = torch.stack(scores).squeeze(-1)
        
        # 选择最高分动作
        best_idx = torch.argmax(scores).item()
        best_action = candidates[best_idx]
        best_score = scores[best_idx]
        
        return best_action, best_score
    
    def _generate_candidates(self, hand_cards: List[int]) -> List[List[int]]:
        """生成所有可能的出牌组合"""
        candidates = []
        n = len(hand_cards)
        
        # 单张
        for i in range(n):
            candidates.append([hand_cards[i]])
        
        # 对子
        rank_groups = {}
        for card in hand_cards:
            rank = CARD_RANK[card]
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        
        for rank, cards in rank_groups.items():
            if len(cards) >= 2:
                # 生成所有对子组合
                for i in range(len(cards)):
                    for j in range(i + 1, len(cards)):
                        candidates.append([cards[i], cards[j]])
        
        # 三张
        for rank, cards in rank_groups.items():
            if len(cards) >= 3:
                # 生成所有三张组合
                from itertools import combinations
                for combo in combinations(cards, 3):
                    candidates.append(list(combo))
        
        # 5张牌型（顺子、同花、葫芦等）
        five_card_candidates = self._generate_five_card_types(hand_cards)
        candidates.extend(five_card_candidates)
        
        return candidates
    
    def _generate_five_card_types(self, hand_cards: List[int]) -> List[List[int]]:
        """生成所有可能的5张牌型"""
        candidates = []
        from itertools import combinations
        
        # 从手牌中选5张的所有组合
        if len(hand_cards) >= 5:
            for combo in combinations(hand_cards, 5):
                combo = list(combo)
                if self._is_valid_five_type(combo):
                    candidates.append(combo)
        
        return candidates
    
    def _is_valid_five_type(self, cards: List[int]) -> bool:
        """检查是否为有效的5张牌型"""
        ranks = sorted([CARD_RANK[c] for c in cards])
        suits = [CARD_SUIT[c] for c in cards]
        
        # 五张王 (rank 16=小王, 17=大王)
        if all(r >= 16 for r in ranks):
            return True
        
        # 炸弹×5
        if len(set(ranks)) == 1:
            return True
        
        # 同花顺
        is_flush = len(set(suits)) == 1
        is_straight = ranks == list(range(ranks[0], ranks[0] + 5))
        if is_flush and is_straight:
            return True
        
        # 4带1
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        if sorted(rank_counts.values()) == [1, 4]:
            return True
        
        # 葫芦
        if sorted(rank_counts.values()) == [2, 3]:
            return True
        
        # 烂同花
        if is_flush:
            return True
        
        # 杂顺
        if is_straight:
            return True
        
        return False


class CardPlayTrainer:
    """出牌模块训练器"""
    
    def __init__(self, hidden_dim=256, lr=1e-3):
        self.model = CardPlayModule(hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def compute_reward(self, action: List[int], hand_before: List[int], 
                       hand_after: List[int], game_result: dict = None) -> float:
        """
        计算出牌动作的奖励 - 优化版
        核心目标：手数最少 + 牌力合理 + 小牌优先
        """
        reward = 0.0
        n_cards = len(action)
        
        # ========== 1. 手牌减少奖励（核心）==========
        # 鼓励一次出多张，减少总手数
        # 梯度设计：1张0.01 < 2张0.03 < 3张0.05 < 5张0.10
        card_reduction_reward = {1: 0.01, 2: 0.03, 3: 0.05, 5: 0.10}.get(n_cards, 0.0)
        reward += card_reduction_reward
        
        # ========== 2. 小牌优先奖励 ==========
        action_ranks = [CARD_RANK[c] for c in action]
        hand_ranks = [CARD_RANK[c] for c in hand_before]
        min_hand_rank = min(hand_ranks) if hand_ranks else 2
        max_action_rank = max(action_ranks) if action_ranks else 17
        
        # 出的是当前手牌中最小的牌，大奖励
        if max_action_rank == min_hand_rank:
            reward += 0.05  # 出最小牌
        elif max_action_rank <= min_hand_rank + 2:
            reward += 0.03  # 出较小牌
        elif max_action_rank <= min_hand_rank + 4:
            reward += 0.01  # 出中等牌
        # 出大牌不给奖励（应该留着回手）
        
        # ========== 3. 剩余手牌质量奖励 ==========
        # 评估剩余手牌能否组成好的组合
        remaining_quality = self._evaluate_hand_quality(hand_after)
        reward += remaining_quality * 0.05
        
        # ========== 4. 牌型选择奖励 ==========
        # 鼓励选择更优的5张牌型
        if n_cards == 5:
            five_type = self._classify_five_type_simple(action)
            type_bonus = {
                6: 0.03,  # 五张王
                5: 0.03,  # 炸弹
                4: 0.02,  # 同花顺
                3: 0.01,  # 4带1
                2: 0.01,  # 葫芦
                1: 0.0,   # 烂同花
                0: 0.0    # 杂顺
            }.get(five_type, 0.0)
            reward += type_bonus
        
        # ========== 5. 终局奖励（如果有）==========
        if game_result:
            finish_rank = game_result.get('finish_rank', -1)
            if finish_rank == 0:  # 头游
                reward += 1.0
            elif finish_rank == 1:  # 二游
                reward += 0.5
            elif finish_rank == 2:  # 三游
                reward += 0.2
        
        return reward
    
    def _evaluate_hand_quality(self, hand: List[int]) -> float:
        """评估手牌质量（0-1，越高越好）"""
        if not hand:
            return 1.0  # 没牌了，最好
        
        n = len(hand)
        
        # 按rank分组
        rank_groups = {}
        for card in hand:
            rank = CARD_RANK[card]
            rank_groups[rank] = rank_groups.get(rank, 0) + 1
        
        # 计算潜在组合数
        potential_plays = 0
        
        # 单张
        potential_plays += n
        
        # 对子
        for count in rank_groups.values():
            if count >= 2:
                potential_plays += 1
        
        # 三张
        for count in rank_groups.values():
            if count >= 3:
                potential_plays += 1
        
        # 5张牌型（简化估计）
        if n >= 5:
            # 估计能组成多少5张牌型
            potential_plays += n // 5
        
        # 质量 = 潜在出牌数 / 手牌数（越高说明组合越丰富）
        quality = min(potential_plays / n, 2.0) / 2.0  # 归一化到0-1
        
        return quality
    
    def _classify_five_type_simple(self, cards: List[int]) -> int:
        """简化版5张牌型分类，返回类型编号"""
        ranks = sorted([CARD_RANK[c] for c in cards])
        suits = [CARD_SUIT[c] for c in cards]
        
        # 五张王 (rank 16=小王, 17=大王)
        if all(r >= 16 for r in ranks):
            return 6
        
        # 炸弹
        if len(set(ranks)) == 1:
            return 5
        
        # 同花顺
        is_flush = len(set(suits)) == 1
        is_straight = ranks == list(range(ranks[0], ranks[0] + 5))
        if is_flush and is_straight:
            return 4
        
        # 4带1
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        if sorted(rank_counts.values()) == [1, 4]:
            return 3
        
        # 葫芦
        if sorted(rank_counts.values()) == [2, 3]:
            return 2
        
        # 烂同花
        if is_flush:
            return 1
        
        # 杂顺
        return 0
    
    def train_step(self, hand_cards: List[int], action: List[int], reward: float):
        """单步训练 - 监督学习版"""
        self.optimizer.zero_grad()
        
        # 编码手牌
        hand_feat = self.model.hand_encoder(hand_cards)
        
        # 编码实际采取的动作
        action_feat = self.model.action_encoder.encode_action(action, hand_cards)
        
        # 计算该动作的分数
        combined = torch.cat([hand_feat, action_feat], dim=0)
        predicted_score = self.model.scorer(combined)
        
        # 计算损失（让预测分数接近实际奖励）
        target = torch.tensor([reward], dtype=torch.float32)
        loss = F.mse_loss(predicted_score, target)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


if __name__ == "__main__":
    # 简单测试
    module = CardPlayModule()
    
    # 生成一手测试牌
    test_hand = list(range(27))  # 简化测试
    
    action, score = module(test_hand)
    print(f"测试手牌: {len(test_hand)}张")
    print(f"选中出牌: {action} ({len(action)}张)")
    print(f"动作分数: {score.item():.4f}")

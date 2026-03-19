"""
大怪路子出牌模块 V2 - 高性能优化版
针对速度瓶颈进行全面优化：
1. 用MLP替代LSTM（LSTM是顺序计算，难以并行）
2. 全部向量化操作
3. 减少不必要的计算
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


class FastHandEncoder(nn.Module):
    """高性能手牌编码器 - 用MLP替代LSTM"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 牌面等级嵌入 (3-10-J-Q-K-A-2-小王-大王)
        self.rank_embed = nn.Embedding(18, 32)
        # 花色嵌入
        self.suit_embed = nn.Embedding(5, 16)
        
        # 使用MLP替代LSTM - 全并行计算
        self.card_processor = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 聚合所有牌的信息（用MaxPool替代LSTM）
        self.aggregator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # 统计特征
        self.stats_fc = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 组合特征
        self.combine_fc = nn.Sequential(
            nn.Linear(128 + 64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, hand_indices):
        """
        hand_indices: [batch_size, 27] 手牌索引列表
        全部向量化，无Python循环
        """
        batch_size = hand_indices.size(0)
        
        # ===== 向量化编码所有牌 =====
        flat_indices = hand_indices.view(-1).long()
        
        # 计算rank和suit
        ranks = (flat_indices // 4).clamp(0, 17).long()
        suits = (flat_indices % 4).clamp(0, 4).long()
        
        # 批量嵌入
        rank_embs = self.rank_embed(ranks)  # [batch_size * 27, 32]
        suit_embs = self.suit_embed(suits)  # [batch_size * 27, 16]
        
        # 合并并处理每张牌
        card_embs = torch.cat([rank_embs, suit_embs], dim=-1)  # [batch_size * 27, 48]
        card_processed = self.card_processor(card_embs)  # [batch_size * 27, 64]
        
        # 恢复形状并聚合
        cards_tensor = card_processed.view(batch_size, 27, 64)  # [batch_size, 27, 64]
        
        # 使用Max Pooling聚合（比LSTM快10倍+）
        hand_feat = cards_tensor.max(dim=1)[0]  # [batch_size, 64]
        hand_feat = self.aggregator(hand_feat)  # [batch_size, 128]
        
        # ===== 向量化统计特征 =====
        clamped_hand = hand_indices.clamp(0, 26).long()
        stats = torch.zeros(batch_size, 27, device=hand_indices.device)
        ones = torch.ones_like(clamped_hand, dtype=torch.float)
        stats.scatter_add_(1, clamped_hand, ones)
        stats_feat = self.stats_fc(stats)  # [batch_size, 64]
        
        # 组合
        combined = torch.cat([hand_feat, stats_feat], dim=-1)
        return self.combine_fc(combined)


class FastCardPlayModule(nn.Module):
    """高性能出牌决策网络"""
    
    def __init__(self, hidden_dim=256, action_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 编码器
        self.hand_encoder = FastHandEncoder(hidden_dim)
        
        # 上下文融合 - 简化网络结构
        self.context_fc = nn.Sequential(
            nn.Linear(832, 512),  # 手牌(256) + 动作(512) + 状态(64)
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, hand_indices, action_candidates, game_state):
        """
        hand_indices: [batch_size, 27] 手牌
        action_candidates: [batch_size, num_actions, action_dim] 候选动作
        game_state: [batch_size, 64] 游戏状态
        
        返回: values [batch_size, num_actions]
        """
        batch_size = hand_indices.size(0)
        num_actions = action_candidates.size(1)
        
        # 编码手牌
        hand_feat = self.hand_encoder(hand_indices)  # [batch_size, hidden_dim]
        hand_feat_expanded = hand_feat.unsqueeze(1).expand(-1, num_actions, -1)
        
        # 扩展游戏状态
        state_expanded = game_state.unsqueeze(1).expand(-1, num_actions, -1)
        
        # 融合上下文
        context_input = torch.cat([hand_feat_expanded, action_candidates, state_expanded], dim=-1)
        
        # 使用view优化大矩阵乘法
        context_flat = context_input.view(batch_size * num_actions, -1)
        context_feat_flat = self.context_fc(context_flat)
        context_feat = context_feat_flat.view(batch_size, num_actions, self.hidden_dim)
        
        # 预测价值
        values = self.value_head(context_feat).squeeze(-1)  # [batch_size, num_actions]
        
        return values


class OptimizedRewardCalculator:
    """优化版奖励计算器 - 细粒度奖励设计"""
    
    def __init__(self):
        self.big_cards = {'2', '小王', '大王'}
        
    def calculate_reward(self, state, action, next_state, is_terminal=False):
        """计算细粒度奖励"""
        reward = 0.0
        
        # 1. 基础合规奖励
        reward += self._compliance_reward(action)
        
        # 2. 组牌策略奖励
        reward += self._combination_reward(state, action, next_state)
        
        # 3. 留牌规划奖励
        reward += self._card_preservation_reward(next_state)
        
        # 4. 轮次压制奖励
        reward += self._round_control_reward(action)
        
        # 5. 团队配合奖励
        reward += self._teamwork_reward(state, action, next_state)
        
        # 6. 终局奖励
        if is_terminal:
            reward += self._terminal_reward(next_state)
        
        # 7. 惩罚过度出牌
        reward += self._overkill_penalty(action)
        
        return np.clip(reward, -15, 15)
    
    def _compliance_reward(self, action):
        """基础合规奖励"""
        reward = 0.0
        
        if action.get('is_valid', False):
            card_count = action.get('card_count', 0)
            if card_count in [1, 2, 3, 5]:
                reward += 0.1
            else:
                reward -= 1.5
                if card_count == 4:
                    reward -= 0.5
        else:
            reward -= 1.5
            
        return reward
    
    def _combination_reward(self, state, action, next_state):
        """组牌策略奖励"""
        reward = 0.0
        
        if action.get('is_optimal_combination', False):
            reward += 0.3
        
        if action.get('split_big_combination_unnecessarily', False):
            reward -= 0.4
            
        if action.get('card_count') == 5:
            five_type = action.get('five_card_type', '')
            type_rewards = {
                'five_kings': 0.5, 'five_bomb': 0.4, 'straight_flush': 0.35,
                'four_with_one': 0.25, 'full_house': 0.25,
                'flush': 0.15, 'straight': 0.1
            }
            reward += type_rewards.get(five_type, 0.0)
                
        return reward
    
    def _card_preservation_reward(self, next_state):
        """留牌规划奖励"""
        reward = 0.0
        remaining_cards = next_state.get('remaining_cards', [])
        if not remaining_cards:
            return reward
            
        big_card_count = sum(1 for card in remaining_cards 
                            if any(bc in str(card) for bc in self.big_cards))
        big_card_ratio = big_card_count / len(remaining_cards) if remaining_cards else 0
        
        if 0.2 <= big_card_ratio <= 0.4:
            reward += 0.2
        elif big_card_ratio > 0.5:
            reward -= 0.3
        elif big_card_ratio < 0.1 and len(remaining_cards) > 5:
            reward -= 0.2
            
        return reward
    
    def _round_control_reward(self, action):
        """轮次压制奖励"""
        reward = 0.0
        
        if action.get('is_win_round', False):
            if action.get('use_small_card_win_round', False):
                reward += 0.8
            else:
                reward += 0.5
                
        return reward
    
    def _teamwork_reward(self, state, action, next_state):
        """团队配合奖励"""
        reward = 0.0
        teammate_remain = next_state.get('teammate_cards', 0)
        
        if teammate_remain < 5 and action.get('help_teammate_out', False):
            reward += 1.0
        if teammate_remain < 3 and not action.get('is_win_round', False):
            reward += 0.3
            
        return reward
    
    def _terminal_reward(self, next_state):
        """终局奖励"""
        return 12.0 if next_state.get('my_team_win', False) else -12.0
    
    def _overkill_penalty(self, action):
        """惩罚过度出牌"""
        reward = 0.0
        if action.get('overkill_card', False):
            reward -= 1.0
        if action.get('could_pass_but_played_big', False):
            reward -= 0.8
        return reward


class CosineAnnealingWarmup:
    """线性预热 + 余弦退火学习率调度"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, base_lr=0.0003):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            lr = self.min_lr + (self.base_lr - self.min_lr) * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class FastCardPlayTrainer:
    """高性能训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.reward_calculator = OptimizedRewardCalculator()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
        self.scheduler = None
        
        self.history = {
            'losses': [], 'rewards': [], 'learning_rates': [],
            'compliance_rate': [], 'combination_rate': [], 'win_round_rate': []
        }
        
    def setup_scheduler(self, warmup_epochs, total_epochs):
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer, warmup_epochs, total_epochs, min_lr=1e-6, base_lr=0.0003
        )
        
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            hand_indices = batch['hand_indices'].to(self.device)
            action_candidates = batch['action_candidates'].to(self.device)
            game_state = batch['game_state'].to(self.device)
            target_values = batch['target_values'].to(self.device)
            
            # 前向传播
            values = self.model(hand_indices, action_candidates, game_state)
            predicted_value = values.mean(dim=1, keepdim=True)
            loss = nn.MSELoss()(predicted_value, target_values)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_size = hand_indices.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        current_lr = self.scheduler.step() if self.scheduler else 0.0003
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        self.history['losses'].append(avg_loss)
        self.history['learning_rates'].append(current_lr)
        
        return {'loss': avg_loss, 'lr': current_lr, 'compliance_rate': 0, 'combination_rate': 0}
    
    def save_checkpoint(self, path, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def create_fast_card_play_model(device='cpu'):
    """创建高性能出牌模型"""
    model = FastCardPlayModule(hidden_dim=256, action_dim=512)
    return model.to(device)

"""
大怪路子多样化对手策略
用于训练时防止过拟合
"""
import random
import numpy as np
from collections import defaultdict
from game_env import recognize, TYPE_ORDER, can_beat, TEAM_MAP


class RandomPlayer:
    """随机策略：从合法动作中随机选择"""
    
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        return random.choice(actions)


class GreedyPlayer:
    """贪心策略：出最大的能压过的牌"""
    
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        # 分离 pass 和出牌动作
        play_actions = [a for a in actions if a]
        if not play_actions:
            return []
        
        # 按牌力排序，选最大的
        scored = [(a, self._evaluate(a, obs['trump_rank'])) for a in play_actions]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[0][0]
    
    def _evaluate(self, cards, trump_rank=None):
        """评估牌力"""
        if not cards:
            return -100
        ct = recognize(cards, trump_rank)
        if not ct:
            return -50
        # 牌型优先级 * 100 + 主牌点数
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


class ConservativePlayer:
    """保守策略：出最小的牌，保留大牌"""
    
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        # 首发时出最小的牌
        if obs.get('current_played') is None or len(obs['current_played']) == 0:
            play_actions = [a for a in actions if a]
            if not play_actions:
                return []
            # 选牌数最多的（快速脱手）
            play_actions.sort(key=lambda a: (-len(a), self._min_rank(a)))
            return play_actions[0]
        
        # 跟牌时出刚好能压过的最小牌
        play_actions = [a for a in actions if a]
        if not play_actions:
            return []
        
        current = obs['current_played']
        beatable = [a for a in play_actions if can_beat(current, a, obs['trump_rank'])]
        
        if beatable:
            # 选最小的能压过的
            beatable.sort(key=lambda a: self._evaluate(a, obs['trump_rank']))
            return beatable[0]
        
        return []  # 不出
    
    def _min_rank(self, cards):
        """最小点数"""
        ranks = [c['rank'] for c in cards if not c.get('is_joker')]
        return min(ranks) if ranks else 100
    
    def _evaluate(self, cards, trump_rank=None):
        """评估牌力"""
        ct = recognize(cards, trump_rank)
        if not ct:
            return 100
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


class AggressivePlayer:
    """激进策略：有牌就出，优先出完"""
    
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        play_actions = [a for a in actions if a]
        if not play_actions:
            return []
        
        # 优先出张数多的牌型
        play_actions.sort(key=len, reverse=True)
        
        # 同样张数，出牌力大的
        max_len = len(play_actions[0])
        same_len = [a for a in play_actions if len(a) == max_len]
        if len(same_len) > 1:
            same_len.sort(key=lambda a: self._evaluate(a, obs['trump_rank']), reverse=True)
            return same_len[0]
        
        return play_actions[0]
    
    def _evaluate(self, cards, trump_rank=None):
        ct = recognize(cards, trump_rank)
        if not ct:
            return -50
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


class TeamAwarePlayer:
    """团队策略：考虑队友和对手状态"""
    
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        play_actions = [a for a in actions if a]
        if not play_actions:
            return []
        
        my_team = obs['my_team']
        hand_sizes = obs['hand_sizes']
        cp = obs['current_player']
        
        # 找队友和对手
        teammates = [s for s in range(6) if TEAM_MAP[s] == my_team and s != cp]
        opponents = [s for s in range(6) if TEAM_MAP[s] != my_team]
        
        # 检查队友状态
        teammate_needs_help = False
        teammate_almost_done = False
        
        for t in teammates:
            if hand_sizes[t] <= 3:
                teammate_needs_help = True
            if hand_sizes[t] <= 1:
                teammate_almost_done = True
        
        # 检查对手状态
        opponent_almost_done = any(hand_sizes[o] <= 2 for o in opponents)
        
        # 策略选择
        if obs.get('current_played') is None or len(obs['current_played']) == 0:
            # 首发
            if teammate_almost_done:
                # 队友快出完，出小牌让队友接
                play_actions.sort(key=lambda a: self._evaluate(a, obs['trump_rank']))
                return play_actions[0]
            else:
                # 正常出牌，出大牌型
                play_actions.sort(key=lambda a: (len(a), self._evaluate(a, obs['trump_rank'])), reverse=True)
                return play_actions[0]
        
        else:
            # 跟牌
            current = obs['current_played']
            beatable = [a for a in play_actions if can_beat(current, a, obs['trump_rank'])]
            
            if not beatable:
                return []
            
            if opponent_almost_done:
                # 对手快出完，用大牌压
                beatable.sort(key=lambda a: self._evaluate(a, obs['trump_rank']), reverse=True)
                return beatable[0]
            elif teammate_needs_help:
                # 队友需要帮助，出能压过的最小牌
                beatable.sort(key=lambda a: self._evaluate(a, obs['trump_rank']))
                return beatable[0]
            else:
                # 正常情况，中等力度
                beatable.sort(key=lambda a: self._evaluate(a, obs['trump_rank']))
                mid = len(beatable) // 2
                return beatable[mid] if beatable else []
    
    def _evaluate(self, cards, trump_rank=None):
        ct = recognize(cards, trump_rank)
        if not ct:
            return -50
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


class SmartPlayer:
    """智能策略：综合多种因素"""
    
    def __init__(self, aggressiveness=0.5):
        self.aggressiveness = aggressiveness  # 0=保守，1=激进
    
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        play_actions = [a for a in actions if a]
        if not play_actions:
            return []
        
        my_team = obs['my_team']
        hand_sizes = obs['hand_sizes']
        cp = obs['current_player']
        my_hand_size = hand_sizes[cp]
        
        # 计算局势分数
        teammates = [s for s in range(6) if TEAM_MAP[s] == my_team and s != cp]
        opponents = [s for s in range(6) if TEAM_MAP[s] != my_team]
        
        # 队伍优势度：队友平均手牌少 = 优势
        team_avg = np.mean([hand_sizes[t] for t in teammates]) if teammates else 27
        opp_avg = np.mean([hand_sizes[o] for o in opponents])
        advantage = (opp_avg - team_avg) / 27.0
        
        # 根据优势调整策略
        if advantage > 0.3:
            # 优势大，保守出牌
            style = 'conservative'
        elif advantage < -0.3:
            # 劣势，激进出牌
            style = 'aggressive'
        else:
            # 均势，根据 aggressiveness 参数
            style = 'balanced'
        
        if obs.get('current_played') is None or len(obs['current_played']) == 0:
            # 首发：根据手牌量决定
            if my_hand_size <= 5:
                # 快出完，出大牌
                play_actions.sort(key=lambda a: self._evaluate(a, obs['trump_rank']), reverse=True)
            else:
                # 正常，出中等牌
                play_actions.sort(key=lambda a: self._evaluate(a, obs['trump_rank']))
                mid = len(play_actions) // 3
                play_actions = play_actions[mid:]
            
            return play_actions[0] if play_actions else []
        
        else:
            # 跟牌
            current = obs['current_played']
            beatable = [a for a in play_actions if can_beat(current, a, obs['trump_rank'])]
            
            if not beatable:
                return []
            
            beatable.sort(key=lambda a: self._evaluate(a, obs['trump_rank']))
            
            if style == 'conservative':
                # 出刚好能压过的最小牌
                return beatable[0]
            elif style == 'aggressive':
                # 出大牌压制
                return beatable[-1]
            else:
                # 中等力度
                idx = int(len(beatable) * self.aggressiveness)
                return beatable[min(idx, len(beatable) - 1)]
    
    def _evaluate(self, cards, trump_rank=None):
        ct = recognize(cards, trump_rank)
        if not ct:
            return -50
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


# ─── 对手池 ─────────────────────────────────────────
class OpponentPool:
    """对手池：训练时采样不同风格的对手"""
    
    def __init__(self):
        self.opponents = {
            'random': RandomPlayer(),
            'greedy': GreedyPlayer(),
            'conservative': ConservativePlayer(),
            'aggressive': AggressivePlayer(),
            'team_aware': TeamAwarePlayer(),
            'smart_30': SmartPlayer(aggressiveness=0.3),
            'smart_50': SmartPlayer(aggressiveness=0.5),
            'smart_70': SmartPlayer(aggressiveness=0.7),
        }
    
    def get(self, name):
        return self.opponents.get(name, RandomPlayer())
    
    def sample(self, training_progress=0.0):
        """根据训练进度采样对手"""
        if training_progress < 0.2:
            # 早期：简单对手
            weights = {'random': 0.5, 'greedy': 0.3, 'conservative': 0.2}
        elif training_progress < 0.5:
            # 中期：中等对手
            weights = {
                'conservative': 0.2, 'aggressive': 0.2,
                'team_aware': 0.3, 'smart_50': 0.3
            }
        else:
            # 后期：强对手
            weights = {
                'team_aware': 0.3, 'smart_50': 0.3,
                'smart_70': 0.2, 'smart_30': 0.2
            }
        
        names = list(weights.keys())
        probs = [weights[n] for n in names]
        return self.opponents[random.choices(names, weights=probs)[0]]
    
    def sample_team(self, team_size=3, training_progress=0.0):
        """为一个队伍采样对手策略"""
        return [self.sample(training_progress) for _ in range(team_size)]


# ─── 测试 ───────────────────────────────────────────
if __name__ == '__main__':
    pool = OpponentPool()
    
    print("可用对手策略:")
    for name, player in pool.opponents.items():
        print(f"  - {name}: {player.__class__.__name__}")
    
    print("\n采样测试 (progress=0.0):")
    for _ in range(5):
        p = pool.sample(0.0)
        print(f"  {p.__class__.__name__}")
    
    print("\n采样测试 (progress=0.6):")
    for _ in range(5):
        p = pool.sample(0.6)
        print(f"  {p.__class__.__name__}")

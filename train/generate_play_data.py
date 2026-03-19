"""
生成出牌模块训练数据
使用蒙特卡洛模拟 + 规则策略生成高质量出牌数据
"""

import random
import json
import time
import torch
from typing import List, Dict, Tuple
from game_env import DaguaiEnv as GameEnv, TEAM_MAP, legal_actions
from card_play_module import CardPlayModule, CardPlayTrainer


class RuleBasedPlayer:
    """规则AI：用于生成训练数据"""
    
    def __init__(self, strategy="greedy"):
        self.strategy = strategy
        
    def select_action(self, hand: List[int], legal_actions: List[List[int]], 
                      current_played: List[int], is_first: bool) -> List[int]:
        """
        根据策略选择动作
        strategy: greedy(贪心) / random(随机) / small_first(小牌优先)
        """
        if not legal_actions:
            return []
        
        if self.strategy == "random":
            return random.choice(legal_actions)
        
        elif self.strategy == "greedy":
            # 优先出张数多的
            return max(legal_actions, key=len)
        
        elif self.strategy == "small_first":
            # 小牌优先，但尽量组合
            return self._small_first_strategy(hand, legal_actions, is_first)
        
        return random.choice(legal_actions)
    
    def _small_first_strategy(self, hand: List[int], legal_actions: List[List[int]], 
                              is_first: bool) -> List[int]:
        """小牌优先策略"""
        if not legal_actions:
            return []
        
        # 按张数分组
        by_size = {}
        for action in legal_actions:
            n = len(action)
            if n not in by_size:
                by_size[n] = []
            by_size[n].append(action)
        
        # 优先顺序：5张 > 3张 > 2张 > 1张（减少手数）
        for size in [5, 3, 2, 1]:
            if size in by_size:
                # 同张数里选rank最小的
                actions = by_size[size]
                return min(actions, key=lambda a: max(c // 4 for c in a))
        
        return legal_actions[0]


def simulate_game(env: GameEnv, players: List[RuleBasedPlayer], max_steps: int = 500) -> Dict:
    """
    模拟一局游戏，记录每个出牌决策
    return: {
        "trajectories": [
            {"hand": [...], "action": [...], "reward": float},
            ...
        ],
        "winner": int,
        "steps": int
    }
    """
    obs = env.reset()
    trajectories = [[] for _ in range(6)]  # 每个玩家的轨迹
    step_count = 0
    
    while not env.done and step_count < max_steps:
        cp = env.current_player
        hand = env.hands[cp]
        
        # 获取合法动作
        legal = legal_actions(
            env.hands[cp],
            env.current_played,
            env.trump_rank,
            env.is_first_play
        )
        
        # 规则AI选择动作
        action = players[cp].select_action(
            hand, legal, env.current_played, env.is_first_play
        )
        
        # 记录决策前的状态
        if action:  # 只记录出牌，不记录pass
            trajectories[cp].append({
                "hand": hand.copy(),
                "action": action.copy(),
                "hand_size_before": len(hand)
            })
        
        # 执行动作
        obs, rewards, done, _ = env.step(action)
        step_count += 1
        
        # 更新奖励（游戏结束后统一计算）
        if action:
            trajectories[cp][-1]["immediate_reward"] = rewards[cp]
    
    # 计算最终奖励（根据完赛顺序）
    final_rewards = _compute_final_rewards(env.finish_order)
    
    # 给每个决策添加最终奖励
    for seat in range(6):
        finish_rank = env.finish_order.index(seat) if seat in env.finish_order else -1
        for traj in trajectories[seat]:
            traj["final_reward"] = final_rewards[seat]
            traj["finish_rank"] = finish_rank
            # 综合奖励 = 即时奖励*0.6 + 最终奖励*0.4（让每步决策都考虑终局）
            traj["total_reward"] = traj.get("immediate_reward", 0) * 0.6 + final_rewards[seat] * 0.4
    
    return {
        "trajectories": trajectories,
        "finish_order": env.finish_order,
        "steps": step_count
    }


def _compute_final_rewards(finish_order: List[int]) -> List[float]:
    """根据完赛顺序计算最终奖励"""
    rewards = [0.0] * 6
    
    # 头游 +1.0, 二游 +0.6, 三游 +0.3, 四游 +0.1, 其他 0
    reward_map = {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.1}
    
    for rank, seat in enumerate(finish_order):
        if rank in reward_map:
            rewards[seat] = reward_map[rank]
    
    return rewards


def generate_dataset(n_games: int = 10000, output_file: str = "play_data.json"):
    """生成训练数据集"""
    
    env = GameEnv()
    
    # 混合策略的玩家
    player_strategies = [
        "small_first",  # 玩家0：小牌优先
        "greedy",       # 玩家1：贪心
        "small_first",  # 玩家2
        "random",       # 玩家3：随机
        "small_first",  # 玩家4
        "greedy"        # 玩家5
    ]
    
    all_data = []
    
    print(f"开始生成 {n_games} 局游戏数据...")
    
    for game_idx in range(n_games):
        players = [RuleBasedPlayer(s) for s in player_strategies]
        result = simulate_game(env, players)
        
        # 收集所有玩家的出牌决策
        for seat in range(6):
            for traj in result["trajectories"][seat]:
                all_data.append({
                    "hand": traj["hand"],
                    "action": traj["action"],
                    "reward": traj["total_reward"],
                    "hand_size": traj["hand_size_before"],
                    "finish_rank": result["finish_order"].index(seat) if seat in result["finish_order"] else -1
                })
        
        if (game_idx + 1) % 1000 == 0:
            print(f"  已完成 {game_idx + 1}/{n_games} 局，收集 {len(all_data)} 条数据")
    
    # 保存数据
    with open(output_file, 'w') as f:
        json.dump(all_data, f)
    
    print(f"\n数据生成完成！")
    print(f"  总游戏数: {n_games}")
    print(f"  总样本数: {len(all_data)}")
    print(f"  平均每局: {len(all_data)/n_games:.1f} 个出牌决策")
    print(f"  保存路径: {output_file}")
    
    return all_data


def train_card_play_module(data_file: str = "play_data.json", 
                           epochs: int = 50,
                           batch_size: int = 256,
                           samples_per_epoch: int = 50000,
                           lr: float = 1e-3):
    """训练出牌模块 - 极速版"""
    
    # 加载数据
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"加载数据: {len(data)} 条")
    
    # 分析数据分布
    rewards = [s["reward"] for s in data]
    print(f"  奖励分布: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards)/len(rewards):.3f}")
    
    # 按完赛排名分组
    rank_groups = {}
    for s in data:
        r = s.get("finish_rank", -1)
        if r not in rank_groups:
            rank_groups[r] = []
        rank_groups[r].append(s)
    
    print(f"  完赛排名分布:")
    for r in sorted(rank_groups.keys()):
        print(f"    排名{r}: {len(rank_groups[r])}条 ({len(rank_groups[r])/len(data)*100:.1f}%)")
    
    # 初始化模型
    trainer = CardPlayTrainer(hidden_dim=256, lr=lr)
    
    # 使用学习率调度
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=20, gamma=0.5)
    
    print(f"\n开始训练 {epochs} 轮 (lr={lr}, batch={batch_size}, 每轮{samples_per_epoch}样本)...")
    print(f"{'轮数':>8} {'Loss':>10} {'最佳Loss':>10} {'学习率':>12} {'进度':>8} {'速度':>10} {'时间':>8}")
    print("-" * 80)
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        
        # 每轮随机采样部分数据（不遍历全部，加速）
        epoch_data = random.sample(data, min(samples_per_epoch, len(data)))
        
        # 批量训练
        for i in range(0, len(epoch_data), batch_size):
            batch = epoch_data[i:i+batch_size]
            
            batch_loss = 0.0
            for sample in batch:
                hand = sample["hand"]
                action = sample["action"]
                
                # 使用新的奖励计算
                hand_after = [c for c in hand if c not in action]
                game_result = {"finish_rank": sample.get("finish_rank", -1)}
                reward = trainer.compute_reward(action, hand, hand_after, game_result)
                
                loss = trainer.train_step(hand, action, reward)
                batch_loss += loss
            
            total_loss += batch_loss / len(batch)
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        # 学习率调度
        scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # 计算速度和时间
        epoch_time = time.time() - epoch_start
        speed = len(epoch_data) / epoch_time
        elapsed = time.time() - start_time
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(trainer.model.state_dict(), "card_play_model_best.pt")
        
        # 每轮都输出
        progress = (epoch + 1) / epochs * 100
        print(f"[{epoch+1:>6}] {avg_loss:>10.4f} {best_loss:>10.4f} {current_lr:>12.6f} {progress:>7.1f}% {speed:>8.0f}样本/s {elapsed:>7.0f}s")
    
    # 保存最终模型
    torch.save(trainer.model.state_dict(), "card_play_model.pt")
    print(f"\n模型保存:")
    print(f"  最终模型: card_play_model.pt")
    print(f"  最佳模型: card_play_model_best.pt (loss={best_loss:.4f})")
    
    return trainer.model


def evaluate_model(model: CardPlayModule, n_games: int = 100):
    """评估训练好的出牌模块"""
    
    env = GameEnv()
    rule_player = RuleBasedPlayer("small_first")
    
    wins = 0
    total_steps = 0
    
    print(f"\n评估 {n_games} 局...")
    
    for game_idx in range(n_games):
        obs = env.reset()
        step_count = 0
        
        while not env.done and step_count < 500:
            cp = env.current_player
            hand = env.hands[cp]
            legal = env.get_legal_actions(cp)
            
            if cp == 0:  # 我们的AI
                if legal:
                    action, _ = model(hand)
                    # 确保动作合法
                    if action not in legal:
                        action = random.choice(legal) if legal else []
                else:
                    action = []
            else:  # 规则AI
                action = rule_player.select_action(
                    hand, legal, env.current_played, env.is_first_play
                )
            
            obs, rewards, done, _ = env.step(action)
            step_count += 1
        
        # 检查是否头游
        if env.finish_order and env.finish_order[0] == 0:
            wins += 1
        
        total_steps += step_count
    
    print(f"  胜率: {wins/n_games*100:.1f}%")
    print(f"  平均步数: {total_steps/n_games:.0f}")
    
    return wins / n_games


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # 生成数据
        generate_dataset(n_games=10000, output_file="play_data.json")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        # 训练模型
        model = train_card_play_module(data_file="play_data.json", epochs=100)
        
        # 评估
        evaluate_model(model, n_games=100)
    
    else:
        print("用法:")
        print("  python generate_play_data.py generate  # 生成数据")
        print("  python generate_play_data.py train     # 训练模型")

"""
大怪路子 PPO 出牌行为分析脚本

用法：
  python analyze_behavior.py                       # 默认分析 ppo_latest.pt，跑500局
  python analyze_behavior.py --model ppo_ep600000.pt
  python analyze_behavior.py --n 1000              # 跑1000局，更准确

输出内容：
  1. 牌型使用分布（单张/对子/三张/顺子/对顺/三顺/炸弹/王炸 各占比）
  2. pass 频率（被迫 pass vs 主动 pass）
  3. 手牌多时 vs 手牌少时的出牌风格对比
  4. 开局 vs 残局的牌型偏好
  5. 与旧AC的行为差异对比
"""
import os
import argparse
import random
import numpy as np
import torch
from collections import defaultdict

from game_env import (
    DaguaiEnv, encode_state, legal_actions,
    TEAM_MAP, CARD_RANK, CARD_IS_JOKER, CARD_IS_BIG,
    RANK_SMALL_JOKER, RANK_BIG_JOKER
)
from model_ppo import DaguaiPPONet, select_action as ppo_select
from model import DaguaiNet
from model import select_action as ac_select


# ─── 牌型识别 ────────────────────────────────────────────
def classify_action(action, trump_rank):
    """识别出牌牌型，返回字符串标签"""
    n = len(action)
    if n == 0:
        return 'pass'

    jokers = [c for c in action if CARD_IS_JOKER[c]]
    normals = [c for c in action if not CARD_IS_JOKER[c]]

    # 王炸：2张都是王
    if len(jokers) == 2:
        return '王炸'
    # 纯王
    if len(jokers) > 0 and len(normals) == 0:
        return f'王×{len(jokers)}'

    ranks = sorted([int(CARD_RANK[c]) for c in normals])

    # 炸弹：4张或以上同rank
    if len(set(ranks)) == 1 and n >= 4:
        return f'炸弹×{n}'

    # 三张
    if n == 3 and len(set(ranks)) == 1:
        return '三张'

    # 对子
    if n == 2 and len(set(ranks)) == 1:
        return '对子'

    # 单张
    if n == 1:
        return '单张'

    # 顺子类（5张及以上，连续rank）
    if n >= 5:
        unique_ranks = sorted(set(ranks))
        # 检查是否连续
        is_straight = all(unique_ranks[i+1] - unique_ranks[i] == 1
                          for i in range(len(unique_ranks)-1))
        if is_straight:
            # 对顺（每rank两张）
            if len(ranks) == len(unique_ranks) * 2:
                return '对顺'
            # 三顺（每rank三张）
            if len(ranks) == len(unique_ranks) * 3:
                return '三顺'
            # 单顺
            if len(ranks) == len(unique_ranks):
                return '顺子'

    return f'其他×{n}'


# ─── 收集行为数据 ─────────────────────────────────────────
def collect_behavior(net, n_games, device, temperature=0.3, is_ppo=True):
    """跑 n_games 局，收集指定模型的出牌行为统计"""
    env = DaguaiEnv()

    stats = {
        'card_type_count': defaultdict(int),   # 牌型计数
        'pass_count': 0,                        # pass 次数
        'play_count': 0,                        # 出牌次数
        'hand_size_when_play': [],              # 出牌时手牌张数
        'action_size': [],                      # 出牌张数
        'early_types': defaultdict(int),        # 开局（手牌>18张）牌型
        'late_types': defaultdict(int),         # 残局（手牌<=9张）牌型
        'win_count': 0,
        'score_list': [],
        'top1_count': 0,                        # 头游次数
        'last_count': 0,                        # 末游次数
    }

    for game_i in range(n_games):
        obs = env.reset()
        trump_rank = env.trump_rank

        while not env.done:
            cp = obs['current_player']
            hand_size = len(obs['hand'])

            if is_ppo:
                action, _, _, _, _ = ppo_select(net, obs, device,
                                                temperature=temperature, greedy=True)
            else:
                action, _, _ = ac_select(net, obs, device,
                                         temperature=temperature, greedy=True)

            # 统计行为（只统计 seat 0 作为代表，避免重复计算）
            if cp == 0:
                card_type = classify_action(action, trump_rank)
                stats['card_type_count'][card_type] += 1
                stats['hand_size_when_play'].append(hand_size)

                if len(action) == 0:
                    stats['pass_count'] += 1
                else:
                    stats['play_count'] += 1
                    stats['action_size'].append(len(action))
                    if hand_size > 18:
                        stats['early_types'][card_type] += 1
                    elif hand_size <= 9:
                        stats['late_types'][card_type] += 1

            obs, rewards, done, _ = env.step(action)

        # 局结束统计
        finish_order = list(env.finish_order)
        if 0 in finish_order:
            rank = finish_order.index(0)
            if rank == 0:
                stats['top1_count'] += 1
            if rank == 5:
                stats['last_count'] += 1

        winner_team = getattr(env, 'last_winner_team', -1)
        if TEAM_MAP[0] == winner_team:
            stats['win_count'] += 1
        stats['score_list'].append(getattr(env, 'last_score', 0))

    return stats


# ─── 打印报告 ─────────────────────────────────────────────
def print_report(stats, n_games, label):
    total_actions = sum(stats['card_type_count'].values())
    win_rate = stats['win_count'] / n_games
    avg_score = np.mean(stats['score_list'])
    top1_rate = stats['top1_count'] / n_games
    last_rate = stats['last_count'] / n_games

    print(f"\n{'='*60}")
    print(f"  【{label}】行为分析报告  ({n_games}局)")
    print(f"{'='*60}")

    print(f"\n📊 整体战绩")
    print(f"  胜率:      {win_rate:.1%}")
    print(f"  平均得分:  {avg_score:.2f} / 3")
    print(f"  头游率:    {top1_rate:.1%}")
    print(f"  末游率:    {last_rate:.1%}")

    print(f"\n🃏 牌型使用分布（共{total_actions}次出牌决策）")
    sorted_types = sorted(stats['card_type_count'].items(),
                          key=lambda x: -x[1])
    for ctype, cnt in sorted_types:
        pct = cnt / total_actions * 100
        bar = '█' * int(pct / 2)
        print(f"  {ctype:<10} {cnt:5d}次  {pct:5.1f}%  {bar}")

    pass_total = stats['pass_count'] + stats['play_count']
    if pass_total > 0:
        pass_rate = stats['pass_count'] / pass_total
        print(f"\n⏭️  Pass 频率: {pass_rate:.1%}  "
              f"({stats['pass_count']}次pass / {pass_total}次决策)")

    if stats['action_size']:
        avg_size = np.mean(stats['action_size'])
        print(f"\n📏 平均每次出牌张数: {avg_size:.2f} 张")

    if stats['hand_size_when_play']:
        avg_hand = np.mean(stats['hand_size_when_play'])
        print(f"📦 出牌时平均手牌数: {avg_hand:.1f} 张")

    early_total = sum(stats['early_types'].values())
    if early_total > 0:
        print(f"\n🌅 开局偏好（手牌>18张，共{early_total}次）:")
        for ctype, cnt in sorted(stats['early_types'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {ctype:<10} {cnt/early_total:.1%}")

    late_total = sum(stats['late_types'].values())
    if late_total > 0:
        print(f"\n🌆 残局偏好（手牌≤9张，共{late_total}次）:")
        for ctype, cnt in sorted(stats['late_types'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {ctype:<10} {cnt/late_total:.1%}")


def compare_reports(ppo_stats, ac_stats, n_games):
    """对比PPO和AC的行为差异"""
    print(f"\n{'='*60}")
    print(f"  【PPO vs AC 行为差异对比】")
    print(f"{'='*60}")

    all_types = set(list(ppo_stats['card_type_count'].keys()) +
                    list(ac_stats['card_type_count'].keys()))
    ppo_total = sum(ppo_stats['card_type_count'].values()) or 1
    ac_total  = sum(ac_stats['card_type_count'].values()) or 1

    print(f"\n{'牌型':<10} {'PPO占比':>8} {'AC占比':>8} {'差值':>8}")
    print("-" * 40)
    rows = []
    for ctype in all_types:
        ppo_pct = ppo_stats['card_type_count'][ctype] / ppo_total * 100
        ac_pct  = ac_stats['card_type_count'][ctype] / ac_total * 100
        diff    = ppo_pct - ac_pct
        rows.append((ctype, ppo_pct, ac_pct, diff))
    for ctype, pp, ap, diff in sorted(rows, key=lambda x: -abs(x[3])):
        arrow = '↑' if diff > 1 else ('↓' if diff < -1 else ' ')
        print(f"  {ctype:<10} {pp:6.1f}%   {ap:6.1f}%   {arrow}{abs(diff):4.1f}%")

    # pass 率对比
    ppo_pr = ppo_stats['pass_count'] / (ppo_stats['pass_count'] + ppo_stats['play_count'] + 1)
    ac_pr  = ac_stats['pass_count']  / (ac_stats['pass_count']  + ac_stats['play_count']  + 1)
    print(f"\n  Pass率:  PPO {ppo_pr:.1%}  vs  AC {ac_pr:.1%}")

    ppo_avg = np.mean(ppo_stats['action_size']) if ppo_stats['action_size'] else 0
    ac_avg  = np.mean(ac_stats['action_size'])  if ac_stats['action_size']  else 0
    print(f"  平均出牌张数:  PPO {ppo_avg:.2f}  vs  AC {ac_avg:.2f}")
    if ppo_avg > ac_avg + 0.2:
        print("  → PPO 更倾向于组合出牌（更有策略性）✅")
    elif ppo_avg < ac_avg - 0.2:
        print("  → PPO 更倾向于拆牌出单张（策略性不如AC）⚠️")
    else:
        print("  → PPO 与 AC 出牌风格相近")


# ─── 主入口 ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='PPO模型路径（默认 checkpoints_ppo/ppo_latest.pt）')
    parser.add_argument('--n',    type=int, default=500,
                        help='分析局数（默认500局）')
    parser.add_argument('--temp', type=float, default=0.3,
                        help='出牌温度（默认0.3，贪心模式）')
    parser.add_argument('--no-compare', action='store_true',
                        help='不与AC对比，只分析PPO')
    args = parser.parse_args()

    device = torch.device('cpu')
    base_dir = os.path.dirname(__file__)

    # 加载PPO
    ppo_path = args.model or os.path.join(base_dir, 'checkpoints_ppo', 'ppo_latest.pt')
    if not os.path.exists(ppo_path):
        print(f"❌ PPO模型不存在: {ppo_path}")
        return
    from game_env import FEATURE_DIM
    ppo_net = DaguaiPPONet(hidden_dim=512).to(device)
    ckpt = torch.load(ppo_path, map_location=device)
    ppo_net.load_state_dict(ckpt['model'])
    ppo_ep = ckpt.get('episode', '?')
    ppo_net.eval()
    print(f"✅ PPO模型: {ppo_path}  (ep {ppo_ep})")

    # 加载AC基准
    ac_path = os.path.join(base_dir, 'checkpoints', 'model_ac_baseline.pt')
    ac_net = None
    if not args.no_compare and os.path.exists(ac_path):
        ac_net = DaguaiNet(input_dim=FEATURE_DIM, hidden_dim=512).to(device)
        ckpt_ac = torch.load(ac_path, map_location=device)
        ac_net.load_state_dict(ckpt_ac['model'])
        ac_net.eval()
        print(f"✅ AC基准: {ac_path}")
    else:
        print("⚠️  跳过AC对比")

    print(f"\n开始分析 PPO 行为（{args.n}局）...")
    ppo_stats = collect_behavior(ppo_net, args.n, device, args.temp, is_ppo=True)
    print_report(ppo_stats, args.n, f"PPO ep{ppo_ep}")

    if ac_net is not None:
        print(f"\n开始分析 AC 行为（{args.n}局）...")
        ac_stats = collect_behavior(ac_net, args.n, device, args.temp, is_ppo=False)
        print_report(ac_stats, args.n, "AC基准 ep35000")
        compare_reports(ppo_stats, ac_stats, args.n)

    print("\n✅ 分析完成")


if __name__ == '__main__':
    main()

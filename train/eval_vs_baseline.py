"""
PPO vs AC基准模型 对战评估脚本

用法：
  python eval_vs_baseline.py                        # 默认：最新PPO vs AC基准，跑1000局
  python eval_vs_baseline.py --ppo ppo_ep50000.pt   # 指定PPO checkpoint
  python eval_vs_baseline.py --n 500                # 只跑500局（快速评估）
  python eval_vs_baseline.py --swap                 # 交换座位再跑一次，消除位置偏差

对战配置：
  红队 (seat 0,2,4): PPO 新模型
  蓝队 (seat 1,3,5): AC 旧模型（冻结，不训练）
"""
import os
import argparse
import random
import numpy as np
import torch

from game_env import DaguaiEnv, encode_state, TEAM_MAP
from model_ppo import DaguaiPPONet, select_action as ppo_select
from model import DaguaiNet
from model import select_action as ac_select


def load_ppo(path, device):
    from game_env import FEATURE_DIM
    net = DaguaiPPONet(hidden_dim=512).to(device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        net.load_state_dict(ckpt['model'])
        ep = ckpt.get('episode', '?')
    else:
        net.load_state_dict(ckpt)
        ep = '?'
    net.eval()
    return net, ep


def load_ac(path, device):
    from game_env import FEATURE_DIM
    net = DaguaiNet(input_dim=FEATURE_DIM, hidden_dim=512).to(device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        net.load_state_dict(ckpt['model'])
    else:
        net.load_state_dict(ckpt)
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net


def run_eval(ppo_net, ac_net, device, n_games, ppo_seats, ac_seats, temperature=0.3):
    """
    跑 n_games 局对战。
    ppo_seats: PPO 控制的座位集合（红队或蓝队）
    ac_seats:  AC 控制的座位集合
    """
    env = DaguaiEnv()

    ppo_team  = TEAM_MAP[list(ppo_seats)[0]]  # PPO 所在队编号 (0 or 1)
    wins      = 0
    scores    = []
    top_ranks = []   # PPO队里最高名次（越小越好）

    for _ in range(n_games):
        obs = env.reset()
        while not env.done:
            cp = obs['current_player']
            if cp in ppo_seats:
                action, _, _, _, _ = ppo_select(ppo_net, obs, device,
                                                temperature=temperature, greedy=False)
            elif cp in ac_seats:
                action, _, _ = ac_select(ac_net, obs, device,
                                         temperature=temperature, greedy=False)
            else:
                action = random.choice(obs['legal_actions'])
            obs, rewards, done, _ = env.step(action)

        # 统计
        winner_team = getattr(env, 'last_winner_team', -1)
        score       = getattr(env, 'last_score', 0)
        finish_order = list(env.finish_order)

        if winner_team == ppo_team:
            wins += 1
        scores.append(score)

        # PPO队最高名次
        ppo_ranks = [finish_order.index(s) for s in ppo_seats if s in finish_order]
        if ppo_ranks:
            top_ranks.append(min(ppo_ranks) + 1)  # 1-indexed

    win_rate   = wins / n_games
    avg_score  = np.mean(scores)
    avg_rank   = np.mean(top_ranks) if top_ranks else 0.0
    return win_rate, avg_score, avg_rank


def main():
    parser = argparse.ArgumentParser(description='PPO vs AC基准模型评估')
    parser.add_argument('--ppo',   type=str, default=None,
                        help='PPO 模型路径（默认用 checkpoints_ppo/ppo_latest.pt）')
    parser.add_argument('--ac',    type=str, default=None,
                        help='AC 基准模型路径（默认用 checkpoints/model_ac_baseline.pt）')
    parser.add_argument('--n',     type=int, default=1000,
                        help='对战局数（默认1000）')
    parser.add_argument('--swap',  action='store_true',
                        help='额外交换红蓝座位再跑一次，消除位置偏差')
    parser.add_argument('--temp',  type=float, default=0.3,
                        help='出牌温度（越低越贪心，默认0.3）')
    args = parser.parse_args()

    device = torch.device('cpu')
    base_dir = os.path.dirname(__file__)

    # ── 加载 PPO ──
    ppo_path = args.ppo or os.path.join(base_dir, 'checkpoints_ppo', 'ppo_latest.pt')
    if not os.path.exists(ppo_path):
        print(f"❌ PPO 模型不存在: {ppo_path}")
        return
    ppo_net, ppo_ep = load_ppo(ppo_path, device)
    print(f"✅ PPO 模型: {ppo_path}  (ep {ppo_ep})")

    # ── 加载 AC ──
    ac_path = args.ac or os.path.join(base_dir, 'checkpoints', 'model_ac_baseline.pt')
    if not os.path.exists(ac_path):
        print(f"❌ AC 基准模型不存在: {ac_path}")
        return
    ac_net = load_ac(ac_path, device)
    print(f"✅ AC 基准模型: {ac_path}")

    print(f"\n对战 {args.n} 局，温度={args.temp}...\n")

    # ── 正向对战：PPO=红队(0,2,4)  AC=蓝队(1,3,5) ──
    ppo_seats = {0, 2, 4}
    ac_seats  = {1, 3, 5}
    wr, avg_s, avg_r = run_eval(ppo_net, ac_net, device, args.n,
                                 ppo_seats, ac_seats, args.temp)

    print("=" * 55)
    print(f"  {'指标':<18} {'PPO(红队)':<14} {'AC(蓝队)'}")
    print("-" * 55)
    print(f"  {'胜率':<18} {wr:.1%}          {1-wr:.1%}")
    print(f"  {'平均得分':<16} {avg_s:.2f}/3")
    print(f"  {'PPO最高名次均值':<14} 第{avg_r:.1f}名")
    print("=" * 55)

    if args.swap:
        # ── 交换座位：PPO=蓝队(1,3,5)  AC=红队(0,2,4) ──
        print(f"\n交换座位，再跑 {args.n} 局...\n")
        ppo_seats2 = {1, 3, 5}
        ac_seats2  = {0, 2, 4}
        wr2, avg_s2, avg_r2 = run_eval(ppo_net, ac_net, device, args.n,
                                        ppo_seats2, ac_seats2, args.temp)

        print("=" * 55)
        print(f"  {'指标':<18} {'PPO(蓝队)':<14} {'AC(红队)'}")
        print("-" * 55)
        print(f"  {'胜率':<18} {wr2:.1%}          {1-wr2:.1%}")
        print(f"  {'平均得分':<16} {avg_s2:.2f}/3")
        print(f"  {'PPO最高名次均值':<14} 第{avg_r2:.1f}名")
        print("=" * 55)

        # 综合
        combined_wr = (wr + wr2) / 2
        print(f"\n综合胜率（消除位置偏差）: PPO {combined_wr:.1%}  vs  AC {1-combined_wr:.1%}")


if __name__ == '__main__':
    main()

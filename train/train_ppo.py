"""
大怪路子 PPO 训练脚本
算法：PPO + GAE + 课程学习 + 搭档联合更新
借鉴：DanZero+ 的动作空间处理 + DouZero 的自对弈框架

训练阶段：
  阶段1 (0~20%)  : 3AI + 3随机对手，快速学会基本牌型
  阶段2 (20~50%) : PPO(红队) vs AC旧模型(蓝队)，对抗有策略的对手
  阶段3 (50~100%): 纯自我对弈，精进博弈策略

搭档联合更新：
  同队（seat 0,2,4 或 1,3,5）的梯度合并更新，
  损失函数加入队伍总奖励项，促进配合。
"""
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from game_env import DaguaiEnv, encode_state, TEAM_MAP
from model_ppo import DaguaiPPONet, PPOBuffer, select_action
from model import DaguaiNet  # 旧 AC 网络（作为课程学习阶段2的对手）

# ─── 超参数 ──────────────────────────────────────────
CFG = {
    # 基础
    'lr':               3e-4,
    'total_episodes':   150000,
    'save_interval':    1000,
    'log_interval':     100,
    'hidden_dim':       512,

    # PPO
    'ppo_epochs':       4,          # 每批数据重复训练次数
    'clip_eps':         0.2,        # PPO clip 范围
    'batch_episodes':   16,         # 多少局经验合并一次更新
    'max_steps':        2000,       # 单局最大步数

    # GAE
    'gamma':            0.99,
    'gae_lambda':       0.95,

    # 损失权重
    'value_coef':       0.5,
    'entropy_coef':     0.02,       # 初始熵正则，后期退火
    'team_reward_coef': 0.3,        # 搭档联合奖励权重

    # 探索温度退火
    'temperature':      1.2,
    'min_temperature':  0.3,

    # 课程学习阶段比例
    'stage1_end':       0.2,        # 0~20%: PPO(红) + 随机(蓝)
    'stage2_end':       0.5,        # 20~50%: PPO(红) vs AC旧模型(蓝)
                                    # 50%~: 纯自我对弈
    'ac_baseline_path': 'checkpoints/model_ac_baseline.pt',  # 旧AC模型路径
}


# ─── 课程学习：阶段判断 ───────────────────────────────
def get_stage(episode, total):
    """
    返回当前训练阶段：
      'random' : PPO(红队 0,2,4) vs 随机对手(蓝队 1,3,5)
      'ac'     : PPO(红队 0,2,4) vs AC旧模型(蓝队 1,3,5)
      'self'   : 纯自我对弈（所有座位用 PPO）
    """
    progress = episode / total
    if progress < CFG['stage1_end']:
        return 'random'
    elif progress < CFG['stage2_end']:
        return 'ac'
    else:
        return 'self'


def load_ac_baseline(device):
    """加载旧 AC 模型，冻结权重，只用于推理"""
    from game_env import FEATURE_DIM
    ac_net = DaguaiNet(input_dim=FEATURE_DIM, hidden_dim=512).to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), CFG['ac_baseline_path'])
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # 兼容直接存 state_dict 或包在 dict 里的两种格式
        if isinstance(state, dict) and 'model' in state:
            ac_net.load_state_dict(state['model'])
        else:
            ac_net.load_state_dict(state)
        print(f"✅ AC基准模型加载成功: {ckpt_path}")
    else:
        print(f"⚠️  AC基准模型不存在: {ckpt_path}，阶段2将使用随机对手")
    # 冻结，不参与梯度更新
    for p in ac_net.parameters():
        p.requires_grad_(False)
    ac_net.eval()
    return ac_net


# ─── AC 旧模型动作选择（推理，不记录经验）────────────
def ac_select_action(ac_net, obs, device):
    """用旧 AC 模型选动作，不需要 PPOBuffer 记录"""
    from model import select_action as ac_action
    action, _, _ = ac_action(ac_net, obs, device, temperature=0.5, greedy=False)
    return action


# ─── 单局经验收集 ─────────────────────────────────────
def collect_episode(net, env, device, temperature, stage, ac_net=None):
    """
    跑一局，收集 PPO 模型控制席位的经验。
    stage:
      'random' : 蓝队(1,3,5)使用随机对手
      'ac'     : 蓝队(1,3,5)使用冻结的 AC 旧模型（不记录经验）
      'self'   : 所有席位用 PPO（纯自我对弈）
    返回: buffer (PPOBuffer), final_rewards (list)
    """
    # 蓝队席位
    BLUE_SEATS = {1, 3, 5}

    obs    = env.reset()
    buffer = PPOBuffer()
    steps  = 0

    while not env.done and steps < CFG['max_steps']:
        cp = obs['current_player']
        is_blue = cp in BLUE_SEATS

        if stage == 'random' and is_blue:
            # 随机出牌
            action = random.choice(obs['legal_actions'])
            obs, rewards, done, _ = env.step(action)

        elif stage == 'ac' and is_blue:
            # AC旧模型出牌，不记录经验
            if ac_net is not None:
                action = ac_select_action(ac_net, obs, device)
            else:
                action = random.choice(obs['legal_actions'])
            obs, rewards, done, _ = env.step(action)

        else:
            # PPO 模型出牌，记录经验
            state_feat = encode_state(obs)
            action, log_prob, value, probs, _ = select_action(
                net, obs, device, temperature
            )
            act_vec = np.zeros(54, dtype=np.float32)
            if action:
                from game_env import CARD_VEC_IDX
                idx = CARD_VEC_IDX[action]
                act_vec[idx] = 1.0

            buffer.add(
                state_feat=state_feat,
                act_vec=act_vec,
                log_prob=log_prob,
                value=value,
                reward=0.0,
                seat=cp,
                done=False,
            )
            obs, rewards, done, _ = env.step(action)

            if env.done and len(buffer) > 0:
                buffer.dones[-1] = True

        steps += 1

    if env.done:
        buffer.fill_final_rewards(rewards)

    return buffer, rewards


# ─── PPO 损失计算 ─────────────────────────────────────
def compute_ppo_loss(net, buffer, device, clip_eps, value_coef,
                     entropy_coef, team_reward_coef):
    """
    PPO Clip 损失 + 价值损失 + 熵正则 + 搭档联合奖励
    """
    if len(buffer) == 0:
        return None

    # GAE
    advantages, returns = buffer.compute_gae(CFG['gamma'], CFG['gae_lambda'])

    states     = torch.FloatTensor(np.array(buffer.states)).to(device)
    act_vecs   = torch.FloatTensor(np.array(buffer.act_vecs)).to(device)
    old_lp     = torch.stack(buffer.log_probs).to(device).detach()
    returns_t  = torch.FloatTensor(returns).to(device)
    adv_t      = torch.FloatTensor(advantages).to(device)
    seats_t    = torch.LongTensor(buffer.seats).to(device)

    # 归一化优势
    if adv_t.std() > 1e-8:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # 前向传播
    logits, values = net(states)          # (N,54), (N,)

    # 重算 log_prob
    # 用 act_vec 选出对应 logit：(N,54)·(54,N) → 取对角
    raw_scores = (act_vecs * logits).sum(dim=1)   # (N,)
    act_norms  = act_vecs.sum(dim=1).clamp(min=1.0)
    scores     = raw_scores / act_norms
    # pass 动作 (act_vec 全0) 固定为 -1
    is_pass    = (act_vecs.sum(dim=1) == 0)
    scores     = torch.where(is_pass, torch.full_like(scores, -1.0), scores)
    new_lp     = F.log_softmax(scores.unsqueeze(1), dim=0).squeeze(1)

    # PPO Clip
    ratio      = torch.exp(new_lp - old_lp)
    surr1      = ratio * adv_t
    surr2      = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
    actor_loss = -torch.min(surr1, surr2).mean()

    # 价值损失
    value_loss = F.mse_loss(values, returns_t)

    # 熵正则（用策略分布的熵鼓励探索）
    entropy    = -(new_lp * torch.exp(new_lp)).mean()
    entropy_loss = -entropy   # 最大化熵 = 最小化 -熵

    # 搭档联合奖励：同队 seat 的 returns 均值拉近
    team_loss = torch.tensor(0.0, device=device)
    rewards_t = torch.FloatTensor(buffer.rewards).to(device)
    for team_id in [0, 1]:
        team_mask = torch.tensor(
            [TEAM_MAP[s] == team_id for s in buffer.seats],
            dtype=torch.bool, device=device
        )
        if team_mask.sum() > 1:
            team_returns = returns_t[team_mask]
            # 队友间 returns 方差越小越好（目标一致）
            team_loss = team_loss + team_returns.var()

    total_loss = (actor_loss
                  + value_coef   * value_loss
                  + entropy_coef * entropy_loss
                  + team_reward_coef * team_loss)

    return total_loss, actor_loss.item(), value_loss.item(), entropy.item()


# ─── 主训练循环 ───────────────────────────────────────
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',   type=str,  default=None)
    parser.add_argument('--episodes', type=int,  default=CFG['total_episodes'])
    parser.add_argument('--stage',    type=str,  default=None,
                        help='强制固定训练阶段: random / ac / self')
    args = parser.parse_args()

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ 使用 CUDA 加速")
    else:
        device = torch.device('cpu')
        print("✅ 使用 CPU 训练")

    net       = DaguaiPPONet(hidden_dim=CFG['hidden_dim']).to(device)
    optimizer = Adam(net.parameters(), lr=CFG['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.episodes, eta_min=1e-5)

    # 加载旧 AC 基准模型（阶段2对手）
    ac_net = load_ac_baseline(device)

    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_ppo')
    os.makedirs(save_dir, exist_ok=True)

    start_ep = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_ep = ckpt.get('episode', 0)
        print(f"✅ 从 episode {start_ep} 恢复训练")

    env = DaguaiEnv()
    reward_history = []
    score_history  = []
    win_history    = []
    loss_history   = []
    accumulated    = []
    t0             = time.time()
    last_stage     = None

    print(f"\n开始 PPO 训练，共 {args.episodes} 局...\n")
    print(f"  阶段1 (0~{int(args.episodes*CFG['stage1_end'])}局):  PPO红队 vs 随机蓝队")
    print(f"  阶段2 ({int(args.episodes*CFG['stage1_end'])}~{int(args.episodes*CFG['stage2_end'])}局): PPO红队 vs AC旧模型蓝队")
    print(f"  阶段3 ({int(args.episodes*CFG['stage2_end'])}局~结束): 纯自我对弈\n")

    for ep in range(start_ep, args.episodes):
        progress    = ep / args.episodes
        temperature = max(
            CFG['min_temperature'],
            CFG['temperature'] * (1 - progress * 0.7)
        )
        # 熵系数退火：前期多探索，后期精炼
        entropy_coef = max(0.005, CFG['entropy_coef'] * (1 - progress * 0.5))

        stage = args.stage if args.stage else get_stage(ep, args.episodes)
        # 阶段切换时打印提示
        if stage != last_stage:
            stage_name = {'random': '阶段1: PPO vs 随机',
                          'ac':     '阶段2: PPO vs AC旧模型',
                          'self':   '阶段3: 纯自我对弈'}
            print(f"\n>>> 切换到 {stage_name[stage]} (ep {ep+1})\n")
            last_stage = stage

        # 收集一局
        net.eval()
        buffer, final_rewards = collect_episode(
            net, env, device, temperature, stage,
            ac_net=(ac_net if stage == 'ac' else None)
        )
        accumulated.append(buffer)

        # 统计红队（PPO 控制）
        red_reward = np.mean([final_rewards[s] for s in [0, 2, 4]])
        reward_history.append(red_reward)
        score_history.append(getattr(env, 'last_score', 0))
        win_history.append(1.0 if getattr(env, 'last_winner_team', -1) == 0 else 0.0)

        # 每 batch_episodes 局更新一次
        if (ep + 1) % CFG['batch_episodes'] == 0:
            net.train()
            total_loss_val = 0.0
            n_updates      = 0

            # PPO 多轮更新
            for _ in range(CFG['ppo_epochs']):
                optimizer.zero_grad()
                for buf in accumulated:
                    result = compute_ppo_loss(
                        net, buf, device,
                        clip_eps         = CFG['clip_eps'],
                        value_coef       = CFG['value_coef'],
                        entropy_coef     = entropy_coef,
                        team_reward_coef = CFG['team_reward_coef'],
                    )
                    if result:
                        loss, al, vl, ent = result
                        loss.backward()
                        total_loss_val += loss.item()
                        n_updates      += 1

                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

            scheduler.step()
            if n_updates > 0:
                loss_history.append(total_loss_val / n_updates)
            accumulated = []
            net.eval()

        # 日志
        if (ep + 1) % CFG['log_interval'] == 0:
            elapsed     = time.time() - t0
            avg_reward  = np.mean(reward_history[-CFG['log_interval']:])
            avg_score   = np.mean(score_history[-CFG['log_interval']:])
            win_rate    = np.mean(win_history[-CFG['log_interval']:])
            avg_loss    = np.mean(loss_history[-10:]) if loss_history else 0.0
            eps_per_sec = CFG['log_interval'] / elapsed
            stage_tag   = {'random': '阶段1', 'ac': '阶段2', 'self': '阶段3'}[stage]
            t0 = time.time()

            print(f"[ep {ep+1:>7}] [{stage_tag}] "
                  f"均值奖励={avg_reward:+.3f}  "
                  f"平均得分={avg_score:.2f}/3  "
                  f"胜率={win_rate:.1%}  "
                  f"loss={avg_loss:.4f}  "
                  f"temp={temperature:.2f}  "
                  f"速度={eps_per_sec:.1f}局/s")

        # 保存
        if (ep + 1) % CFG['save_interval'] == 0:
            path = os.path.join(save_dir, f'ppo_ep{ep+1}.pt')
            torch.save({
                'episode':   ep + 1,
                'model':     net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg':       CFG,
            }, path)
            latest = os.path.join(save_dir, 'ppo_latest.pt')
            torch.save({
                'episode':   ep + 1,
                'model':     net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg':       CFG,
            }, latest)
            print(f"  💾 已保存: {path}")

    print("\nPPO 训练完成！")
    print(f"最终胜率: {np.mean(win_history[-1000:]):.1%}")
    print(f"最终平均得分: {np.mean(score_history[-1000:]):.2f} / 3.0")


if __name__ == '__main__':
    train()

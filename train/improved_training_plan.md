# 大怪路子AI训练改进方案

## 一、核心问题诊断

### 1. 状态特征不足
当前224维特征缺少：
- **牌权信息**：谁在控制局面（关键！）
- **出牌历史模式**：对手偏好什么牌型
- **协作信息**：队友需要什么牌
- **进贡记忆**：谁进贡了什么牌

### 2. 动作空间粗糙
- 54维向量无法区分牌型质量
- 同样5张牌，同花顺 >> 顺子，但网络不知道

### 3. 奖励稀疏
- 只有终局奖励（+1/-1）
- 中间决策没有引导信号
- 导致训练收敛慢、策略不稳定

### 4. 对手单一
- 只有随机/固定模型/自我对弈
- 容易过拟合特定风格

---

## 二、改进方案

### 方案A：特征工程增强（最简单，效果明显）

```python
# 改进后的特征维度：224 → 512+
FEATURE_DIM = 54 * 6 + 36 + 24 + 6 + 20

# 1. 手牌特征 (54*4 = 216)
#    - 自己手牌、已出牌、当前桌面牌、合法动作掩码

# 2. 对手牌型统计 (36)  ← 新增
#    - 每个对手已出的牌型分布（单张/对子/三条/顺子/同花...）
#    - 用于预测对手剩余牌型

# 3. 牌权信息 (24)  ← 新增
#    - 当前谁在控制牌权（最近5轮的出牌者）
#    - 每个玩家的"控制力"评分

# 4. 队友协作信号 (6)  ← 新增
#    - 队友是否需要帮助（手牌少、被压制）
#    - 队友是否可以接风

# 5. 将牌/进贡信息 (20)  ← 新增
#    - 当前将牌
#    - 进贡/还贡记录
```

### 方案B：奖励塑形（关键改进）

```python
def compute_shaped_reward(env, action, next_obs):
    """中间奖励 + 终局奖励"""
    reward = 0.0
    
    # 1. 牌权控制奖励（中等重要）
    if action and is_winning_play(env, action):
        reward += 0.1  # 获得牌权
    
    # 2. 出牌效率奖励
    if action:
        cards_out = len(action)
        hand_reduction = env.prev_hand_size - len(env.hands[cp])
        reward += 0.02 * hand_reduction  # 出牌多奖励
    
    # 3. 保护队友奖励（重要！）
    if is_helping_teammate(env, action):
        reward += 0.15  # 帮队友脱困
    
    # 4. 压制对手奖励
    if is_blocking_opponent(env, action):
        reward += 0.1
    
    # 5. 留牌策略奖励（重要！）
    if action and len(env.hands[cp]) > 0:
        remaining_quality = evaluate_hand_quality(env.hands[cp])
        if remaining_quality > 0.7:
            reward += 0.05  # 保留好牌
    
    # 6. 终局奖励（放大）
    if env.done:
        rank = get_rank(cp, env.finish_order)
        reward += [1.5, 0.8, 0.3, -0.2, -0.6, -1.2][rank]
        
        # 队伍胜负额外奖励
        team_bonus = 0.5 if team_wins(cp, env) else -0.3
        reward += team_bonus
    
    return reward
```

### 方案C：对手池多样化

```python
class OpponentPool:
    """多样化对手池，防止过拟合"""
    
    def __init__(self):
        self.opponents = {
            'random': RandomPlayer(),
            'greedy': GreedyPlayer(),      # 出最大牌
            'conservative': ConservativePlayer(),  # 出最小牌
            'aggressive': AggressivePlayer(),  # 有牌就出
            'historical': [],  # 历史模型快照
        }
    
    def sample_opponent(self, training_progress):
        """根据训练进度采样对手"""
        if progress < 0.2:
            # 早期：简单对手
            return random.choice(['random', 'greedy'])
        elif progress < 0.5:
            # 中期：混合对手
            return random.choice(['conservative', 'aggressive', 'historical'])
        else:
            # 后期：强对手 + 自我对弈
            return random.choice(['historical', 'self'])
```

### 方案D：蒙特卡洛树搜索增强（效果最好）

```python
class MCTSPlayer:
    """结合MCTS的AI玩家"""
    
    def __init__(self, net, simulations=100):
        self.net = net
        self.simulations = simulations
    
    def select_action(self, obs):
        # 1. 用神经网络快速评估
        policy, value = self.net.predict(obs)
        
        # 2. 对高概率动作进行MCTS搜索
        top_actions = policy.topk(5)
        
        best_action = None
        best_score = -float('inf')
        
        for action in top_actions:
            # 模拟展开
            score = self.simulate(action, obs)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def simulate(self, action, obs, depth=10):
        """快速模拟到终局"""
        env = clone_env(obs)
        env.step(action)
        
        for _ in range(depth):
            if env.done:
                break
            # 用网络快速选动作
            a = self.net.fast_select(env.get_obs())
            env.step(a)
        
        return self.net.evaluate(env)
```

### 方案E：分离式牌型网络

```python
class CardTypeEncoder(nn.Module):
    """专门编码牌型的子网络"""
    
    def __init__(self):
        super().__init__()
        # 牌型嵌入：9种牌型 + 质量评分
        self.type_embed = nn.Embedding(10, 32)
        self.quality_net = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
    
    def forward(self, cards):
        # 输出：(牌型嵌入, 质量评分)
        card_type = recognize_card_type(cards)
        quality = self.quality_net(cards_to_vec(cards))
        return self.type_embed(card_type), quality


class ImprovedDaguaiNet(nn.Module):
    """改进的网络架构"""
    
    def __init__(self):
        super().__init__()
        # 牌型编码器
        self.type_encoder = CardTypeEncoder()
        
        # 手牌处理（Transformer）
        self.card_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=2
        )
        
        # 玩家关系建模（图注意力）
        self.player_gat = GATConv(64, 32, heads=2)
        
        # 决策头
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 54),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
```

---

## 三、推荐实施路径

### 第一阶段（1-2天）：快速见效
1. **增加奖励塑形**（最重要！）
   - 添加牌权控制奖励
   - 添加保护队友奖励
   - 效果预期：胜率提升10-15%

2. **增强特征**
   - 添加牌权信息
   - 添加对手牌型统计
   - 效果预期：决策更合理

### 第二阶段（3-5天）：架构优化
1. **对手池多样化**
   - 实现多种策略的规则AI
   - 定期保存模型快照
   - 效果预期：泛化能力增强

2. **分离式牌型网络**
   - 用专门的子网络编码牌型
   - 效果预期：牌型判断更准确

### 第三阶段（7天+）：深度优化
1. **MCTS增强**
   - 结合蒙特卡洛树搜索
   - 效果预期：接近人类高手水平

2. **对手建模**
   - 预测对手手牌分布
   - 效果预期：策略更具针对性

---

## 四、具体代码改动

### 改动1：增强特征编码

```python
# game_env.py 中添加

def encode_state_enhanced(obs):
    """增强版状态编码，维度 512"""
    # 原始特征 (224)
    base_feat = encode_state(obs)
    
    # 新增：牌权信息 (24)
    control_feat = np.zeros(24, dtype=np.float32)
    recent_actors = obs.get('recent_actors', [0]*6)
    for i, actor in enumerate(recent_actors[:6]):
        control_feat[i*4 + actor//2] = 1.0  # 6轮 * 4位（2队）
    
    # 新增：对手牌型统计 (36)
    opponent_stats = np.zeros(36, dtype=np.float32)
    for seat, stats in obs.get('opponent_card_types', {}).items():
        for j, count in enumerate(stats.values()):
            opponent_stats[seat * 9 + j] = min(count / 3.0, 1.0)
    
    # 新增：队友状态 (6)
    teammate_feat = np.zeros(6, dtype=np.float32)
    teammate = 4 if obs['my_team'] == 0 else 5  # 简化
    teammate_feat[0] = obs['hand_sizes'][teammate] / 27.0
    teammate_feat[1] = 1.0 if obs['hand_sizes'][teammate] < 5 else 0.0
    
    # 新增：进贡信息 (20)
    tribute_feat = np.zeros(20, dtype=np.float32)
    # ... 进贡历史编码
    
    return np.concatenate([
        base_feat, control_feat, opponent_stats, teammate_feat, tribute_feat
    ])
```

### 改动2：奖励塑形

```python
# train_ppo.py 中添加

def compute_step_reward(env, action, next_obs, done):
    """中间步骤奖励"""
    reward = 0.0
    cp = env.current_player
    
    if not action:  # pass
        return -0.02  # 小惩罚，鼓励出牌
    
    # 出牌效率
    cards_played = len(action)
    reward += 0.01 * cards_played
    
    # 牌权控制（关键！）
    if env.current_played == action:  # 获得牌权
        card_type = recognize(action)
        type_order = TYPE_ORDER.get(card_type[0], 0) if card_type else 0
        reward += 0.05 + 0.02 * type_order  # 大牌型奖励更多
    
    # 保护队友
    teammate = [s for s in [0,2,4] if s != cp] if TEAM_MAP[cp] == 0 else [s for s in [1,3,5] if s != cp]
    for t in teammate:
        if env.hands[t] and len(env.hands[t]) <= 3:
            # 队友牌少，这手牌帮ta脱困
            reward += 0.08
    
    return reward
```

### 改动3：多种对手策略

```python
# opponents.py 新建

class GreedyPlayer:
    """贪心策略：出最大的能压过的牌"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        # 按牌力排序，选最大的
        scored = [(a, self.evaluate(a)) for a in actions if a]
        if not scored:
            return []
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def evaluate(self, cards):
        if not cards:
            return -100
        ct = recognize(cards)
        if not ct:
            return -50
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


class ConservativePlayer:
    """保守策略：出最小的牌，保留大牌"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        # 按牌力排序，选最小的
        scored = [(a, self.evaluate(a)) for a in actions if a]
        if not scored:
            return []
        scored.sort(key=lambda x: x[1])
        return scored[0][0]
    
    def evaluate(self, cards):
        ct = recognize(cards)
        if not ct:
            return 100
        return TYPE_ORDER[ct[0]] * 100 + ct[1]


class AggressivePlayer:
    """激进策略：有牌就出，优先出完"""
    def select_action(self, obs):
        actions = obs['legal_actions']
        if not actions:
            return []
        
        # 优先出张数多的
        play_actions = [a for a in actions if a]
        if not play_actions:
            return []
        play_actions.sort(key=len, reverse=True)
        return play_actions[0]
```

---

## 五、预期效果

| 改进项 | 开发时间 | 胜率提升 | 优先级 |
|--------|----------|----------|--------|
| 奖励塑形 | 1天 | 10-15% | ⭐⭐⭐⭐⭐ |
| 特征增强 | 1天 | 5-10% | ⭐⭐⭐⭐ |
| 对手池 | 2天 | 5-8% | ⭐⭐⭐ |
| 牌型网络 | 3天 | 5-10% | ⭐⭐⭐ |
| MCTS | 5天 | 10-20% | ⭐⭐ |

**推荐先实现：奖励塑形 + 特征增强**，投入产出比最高！

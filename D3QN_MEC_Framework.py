import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

# ==========================================
# 0. 固定随机种子
# ==========================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    def __init__(self):
        self.num_users = 20      
        self.num_services = 40   
        self.cache_capacity = 10 
        self.num_channels = 5    
        
        self.feature_dim = self.num_users * (self.num_services + 1)
        
        # RL 参数
        self.gamma = 0.95
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05       # [修改] 保持 5% 的随机性，不要降到 0.01
        
        # [核心修改] 衰减极慢，确保前 1000 Episode 都在大量探索
        # 0.9995^20000_steps approx 0.05
        self.epsilon_decay = 0.9995   
        
        self.learning_rate = 0.0003   # [修改] 再小一点，求稳
        self.batch_size = 64          # [修改] 小 Batch 更新更频繁
        self.memory_size = 10000      # [修改] 增大 Buffer
        self.target_update = 200      # [修改] Target 网络更新慢一点
        self.hidden_dim_1 = 256
        self.hidden_dim_2 = 256

        # 物理参数
        self.task_input_sizes = np.random.uniform(0.5, 1.5, size=self.num_services) # [微调] 稍微小一点
        self.service_sizes = np.random.randint(1, 4, size=self.num_services)        
        self.task_cycles = self.task_input_sizes * 0.5 
        
        self.time_slot_limit = 0.3    
        self.bandwidth = 20e6         
        self.noise_power = 1e-13      
        self.trans_power_user = 0.5   
        
        self.f_local = 1.0e9          
        self.f_edge = 20.0e9          
        self.f_cloud = 100.0e9        
        self.rate_fiber = 100 * 1e6    

        # [核心修改] 降低罚分，让梯度更平滑
        # 正常能耗 ~0.5, 罚分设为 2.0 足够让 AI 感到“痛”，但不至于梯度爆炸
        self.penalty = 2.0           
        
        self.zeta = 1e-28             
        self.p_exe_edge = 1.0         
        self.game_max_iter = 20       

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()

# ==========================================
# 2. D3QN 模型 (支持 One-Hot)
# ==========================================
class D3QN_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(D3QN_Network, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim_1, cfg.hidden_dim_2),
            nn.ReLU()
        )
        # V(s): 状态价值 (Scalar)
        self.value_stream = nn.Sequential(
            nn.Linear(cfg.hidden_dim_2, 1) 
        )
        # A(s, a): 每个服务的优势值 (Vector, size=num_services)
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg.hidden_dim_2, output_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # 关键修改：分别返回，不要在这里合并，因为合并逻辑取决于动作选择
        return values, advantages

# ==========================================
# 3. 辅助函数：One-Hot 转换
# ==========================================
def to_onehot(state_arr, cfg):
    """
    将 [batch, num_users] 的整数状态转换为 [batch, num_users * (num_services+1)] 的 One-Hot 向量
    """
    if len(state_arr.shape) == 1:
        state_arr = state_arr[np.newaxis, :] # 增加 batch 维度
    
    batch_size = state_arr.shape[0]
    num_classes = cfg.num_services + 1
    
    # 创建 One-Hot 容器
    one_hot = np.zeros((batch_size, cfg.num_users, num_classes), dtype=np.float32)
    
    # 填充
    for i in range(batch_size):
        for u in range(cfg.num_users):
            val = state_arr[i, u]
            one_hot[i, u, int(val)] = 1.0
            
    # Flatten: [batch, num_users * num_classes]
    return one_hot.reshape(batch_size, -1)

# ==========================================
# 4. 背包算法
# ==========================================
def solve_knapsack(q_values, sizes, capacity):
    K = len(q_values)
    C = int(capacity)
    q_values = np.clip(q_values, 0, None) 
    dp = np.zeros((K + 1, C + 1))
    for i in range(1, K + 1):
        w_item = int(sizes[i-1])
        v_item = q_values[i-1]
        for w in range(1, C + 1):
            if w_item <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-w_item] + v_item)
            else:
                dp[i][w] = dp[i-1][w]
    selected = np.zeros(K)
    w = C
    for i in range(K, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected[i-1] = 1
            w -= int(sizes[i-1])
    return selected

# ==========================================
# 5. Agent (修正版：严格背包Target + 梯度控制)
# ==========================================
class D3QNAgent:
    def __init__(self, config):
        self.cfg = config
        self.policy_net = D3QN_Network(config.feature_dim, config.num_services).to(config.device)
        self.target_net = D3QN_Network(config.feature_dim, config.num_services).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # [修改] 降低学习率，防止 Sum Q 导致的梯度爆炸
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
        self.memory = deque(maxlen=config.memory_size)
        self.loss_fn = nn.MSELoss()
        self.epsilon = config.epsilon_start
        self.steps = 0

    def select_action(self, state):
        self.steps += 1
        state_onehot = to_onehot(state, self.cfg)
        
        if random.random() < self.epsilon:
            # 随机探索：随机生成 Advantage 并解背包
            # 这样保证随机出的动作也是符合物理约束的
            random_adv = np.random.uniform(-1, 1, size=self.cfg.num_services)
            action = solve_knapsack(random_adv, self.cfg.service_sizes, self.cfg.cache_capacity)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_onehot).to(self.cfg.device)
                v, a = self.policy_net(state_tensor)
                advantages = a.cpu().numpy()[0]
                # Dueling Centering
                advantages = advantages - advantages.mean()
                action = solve_knapsack(advantages, self.cfg.service_sizes, self.cfg.cache_capacity)
        return action

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.cfg.batch_size)
        state, action, reward, next_state = zip(*batch)
        
        # 数据准备
        state_onehot = to_onehot(np.array(state), self.cfg)
        next_state_onehot = to_onehot(np.array(next_state), self.cfg)
        
        state_t = torch.FloatTensor(state_onehot).to(self.cfg.device)
        action_t = torch.FloatTensor(np.array(action)).to(self.cfg.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.cfg.device)
        next_state_t = torch.FloatTensor(next_state_onehot).to(self.cfg.device)
        
        # --- 1. 计算当前 Q 值 (Current Q) ---
        curr_v, curr_a = self.policy_net(state_t)
        curr_a_centered = curr_a - curr_a.mean(dim=1, keepdim=True)
        
        # AOF: Q = V + Sum(A * mask)
        # 这一步计算的是"Agent实际采取的那个动作组合"的价值
        current_q_value = curr_v + torch.sum(curr_a_centered * action_t, dim=1, keepdim=True)
        
        # --- 2. 计算目标 Q 值 (Target Q) - Double DQN ---
        with torch.no_grad():
            # A. 使用 Policy Net 决定"哪个是最佳动作组合" (Selection)
            # 注意：必须解背包！不能用 TopK 近似！
            next_v_eval, next_a_eval = self.policy_net(next_state_t)
            next_a_eval_np = next_a_eval.cpu().numpy()
            
            # 在 CPU 上循环 Batch 解背包 (这是保证收敛的代价)
            # 虽然慢，但保证了 Target Value 是物理可行的
            best_next_actions = []
            for i in range(self.cfg.batch_size):
                adv = next_a_eval_np[i]
                adv = adv - adv.mean() # Centering
                # 解背包
                act = solve_knapsack(adv, self.cfg.service_sizes, self.cfg.cache_capacity)
                best_next_actions.append(act)
            
            best_next_actions = torch.FloatTensor(np.array(best_next_actions)).to(self.cfg.device)
            
            # B. 使用 Target Net 评估该动作组合的价值 (Evaluation)
            next_v_target, next_a_target = self.target_net(next_state_t)
            next_a_target_centered = next_a_target - next_a_target.mean(dim=1, keepdim=True)
            
            # 计算 Target Q
            max_next_q = next_v_target + torch.sum(next_a_target_centered * best_next_actions, dim=1, keepdim=True)
            
            target_q_value = reward_t + self.cfg.gamma * max_next_q

        loss = self.loss_fn(current_q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # [关键] 强力梯度裁剪，防止 Sum Q 导致的梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        
        self.optimizer.step()
        
        # Epsilon Decay
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==========================================
# 6. 物理环境
# ==========================================
class MECEnvironment:
    def __init__(self, config):
        self.cfg = config
        self.user_locations = np.random.uniform(0, 1000, size=(config.num_users, 2))
        self.bs_location = np.array([500, 500])
        self.distances = np.linalg.norm(self.user_locations - self.bs_location, axis=1)
        self.distances = np.maximum(self.distances, 10.0)
        path_loss_exp = 4.0
        self.gains = self.distances ** (-path_loss_exp)
        self.rx_powers = self.cfg.trans_power_user * self.gains

    def reset(self):
        return self._generate_requests()
    
    def _calculate_cost_for_user(self, u, strategy, current_strategies, caching_decision, task_id):
        # 1. Local
        if strategy == 0:
            S_k = self.cfg.task_cycles[task_id] * 1e9
            t_local = S_k / self.cfg.f_local
            e_local = self.cfg.zeta * (self.cfg.f_local**2) * S_k
            if t_local > self.cfg.time_slot_limit:
                e_local += self.cfg.penalty
            return e_local

        # 2. Offload
        interference = 0
        for v in range(self.cfg.num_users):
            if v != u and current_strategies[v] == strategy:
                interference += self.rx_powers[v]
        
        # [修正] 增加一点基底噪声防止除零或极端数值
        sinr = self.rx_powers[u] / (self.cfg.noise_power + interference + 1e-9)
        rate = self.cfg.bandwidth * np.log2(1 + sinr)
        if rate < 1e3: rate = 1e3 
        
        k = task_id
        I_k = self.cfg.task_input_sizes[k] * 8 * 1e6
        S_k = self.cfg.task_cycles[k] * 1e9
        
        is_cached = (caching_decision[k] == 1)
        
        if is_cached:
            # Edge
            t_trans = I_k / rate
            t_exe = S_k / self.cfg.f_edge
            delay = t_trans + t_exe
            energy = self.cfg.trans_power_user * t_trans + self.cfg.p_exe_edge * t_exe
        else:
            # Cloud (Fiber Bottleneck)
            t_trans_edge = I_k / rate
            t_trans_fiber = I_k / self.cfg.rate_fiber
            t_exe_cloud = S_k / self.cfg.f_cloud
            delay = t_trans_edge + t_trans_fiber + t_exe_cloud
            energy = self.cfg.trans_power_user * (t_trans_edge + t_trans_fiber)
            
        if delay > self.cfg.time_slot_limit:
            energy += self.cfg.penalty
            
        return energy

    def find_nash_equilibrium(self, caching_decision, user_requests):
        strategies = np.zeros(self.cfg.num_users, dtype=int)
        active_users = [u for u in range(self.cfg.num_users) if user_requests[u] > 0]
        
        for it in range(self.cfg.game_max_iter):
            changed = False
            np.random.shuffle(active_users)
            for u in active_users:
                task_id = user_requests[u] - 1
                current_s = strategies[u]
                current_cost = self._calculate_cost_for_user(u, current_s, strategies, caching_decision, task_id)
                best_s = current_s
                min_cost = current_cost
                
                potential_strategies = list(range(self.cfg.num_channels + 1))
                for s in potential_strategies:
                    if s == current_s: continue
                    cost = self._calculate_cost_for_user(u, s, strategies, caching_decision, task_id)
                    if cost < min_cost:
                        min_cost = cost
                        best_s = s
                if best_s != current_s:
                    strategies[u] = best_s
                    changed = True
            if not changed:
                break
        return strategies

    def step(self, action):
        next_state = self._generate_requests()
        # [新增] 计算缓存命中率 (仅供观察)
        # action 是 0/1 向量, next_state 是用户请求
        hits = 0
        total_reqs = 0
        for u in range(self.cfg.num_users):
            req_id = next_state[u]
            if req_id > 0: # 0 是无请求
                total_reqs += 1
                if action[req_id-1] == 1:
                    hits += 1
        hit_rate = hits / (total_reqs + 1e-9)

        # 纳什均衡计算
        final_strategies = self.find_nash_equilibrium(action, next_state)
        
        total_energy = 0
        penalty_count = 0
        for u in range(self.cfg.num_users):
            req = next_state[u]
            if req == 0: continue
            task_id = req - 1
            s = final_strategies[u]
            energy = self._calculate_cost_for_user(u, s, final_strategies, action, task_id)
            
            # [技巧] Reward Clipping: 把能耗缩放到 [-1, 0] 之间，防止数值过大
            # 正常 energy ~0.5 -> -0.5
            # 罚分 energy ~2.0 -> -2.0
            total_energy += energy
            
            if energy > self.cfg.penalty * 0.8: 
                penalty_count += 1

        reward = -total_energy
        # 归一化 Reward，让神经网络更好训练 (除以用户数)
        reward = reward / self.cfg.num_users 

        return next_state, reward, False, penalty_count, hit_rate

    def _generate_requests(self):
        # 强 Zipf，降低学习难度
        probs = 1.0 / (np.arange(1, self.cfg.num_services + 2)**1.5)
        probs = probs / probs.sum()
        return np.random.choice(np.arange(0, self.cfg.num_services + 1), 
                              size=self.cfg.num_users, 
                              p=probs)

# ==========================================
# 7. 主循环
# ==========================================
if __name__ == "__main__":
    env = MECEnvironment(cfg)
    agent = D3QNAgent(cfg)
    
    # 预热 Replay Buffer：先随机跑一会，不训练
    print("正在预热 Replay Buffer...")
    state = env.reset()
    for _ in range(cfg.batch_size * 5):
        action = agent.select_action(state)
        # 这里 step 的返回值要对应上面的修改
        next_state, reward, done, pens, hit_rate = env.step(action)
        agent.store_experience(state, action, reward, next_state)
        state = next_state
        
    episodes = 2000 # 可以跑少一点，因为我们减慢了 decay
    print(f"开始训练 | 罚分: {cfg.penalty} | 衰减: {cfg.epsilon_decay}")
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        total_penalties = 0
        total_hits = 0
        
        for t in range(20): 
            action = agent.select_action(state)
            next_state, reward, done, pens, hit_rate = env.step(action)
            agent.store_experience(state, action, reward, next_state)
            
            loss = agent.update() # 每次 step 都更新
            
            state = next_state
            total_reward += reward
            total_penalties += pens
            total_hits += hit_rate
            
            if agent.steps % cfg.target_update == 0:
                agent.update_target_network()
        
        avg_reward = total_reward / 20
        avg_hit_rate = total_hits / 20
        
        if (e+1) % 20 == 0: # 打印频繁一点
            # 观察 agent 的 epsilon 还有多少
            # 观察 Hit Rate 是否上升 (这比 Reward 更直观)
            print(f"Ep {e+1:4d} | Rew: {avg_reward:.3f} | Pens: {total_penalties:3d} | HitRate: {avg_hit_rate:.2f} | Eps: {agent.epsilon:.3f}")
            
            # [Debug] 每100轮看一眼 Agent 到底想缓存哪些服务 (Top 5 Advantage)
            if (e+1) % 100 == 0:
                with torch.no_grad():
                    test_state = to_onehot(state, cfg)
                    test_tensor = torch.FloatTensor(test_state).to(cfg.device)
                    _, adv = agent.policy_net(test_tensor)
                    adv = adv.cpu().numpy()[0]
                    # 打印 Advantage 最大的前5个服务的 ID
                    top_k_idx = np.argsort(adv)[-10:][::-1]
                    print(f"   >>> 当前模型倾向于缓存的服务 ID: {top_k_idx}")
                    print(f"   >>> (服务 0,1,2... 应该是最热门的，如果包含这些说明学会了)")
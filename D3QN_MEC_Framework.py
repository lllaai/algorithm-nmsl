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
    np.random.seed(seed)
    random.seed(seed)

    # 只在 CUDA 存在时调用
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        # 输入维度
        self.feature_dim = self.num_users * (self.num_services + 1)

        # RL 参数
        self.gamma = 0.95
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.996
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.memory_size = 5000
        self.target_update = 50
        self.hidden_dim_1 = 300
        self.hidden_dim_2 = 400

        self.task_input_sizes = np.random.uniform(0.5, 2.0, size=self.num_services)
        self.service_sizes = np.random.randint(1, 5, size=self.num_services)
        self.task_cycles = self.task_input_sizes * 0.5

        self.time_slot_limit = 0.3
        self.bandwidth = 20e6
        self.noise_power = 1e-13
        self.trans_power_user = 0.5

        self.f_local = 1.0e9
        self.f_edge = 20.0e9
        self.f_cloud = 100.0e9
        self.rate_fiber = 100 * 1e6

        self.penalty = 9.0
        self.zeta = 1e-28
        self.p_exe_edge = 1.0
        self.game_max_iter = 20

        # ==========================================
        # 关键：正确选择 MPS / CUDA / CPU
        # ==========================================
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("Using device:", self.device)


cfg = Config()


# ==========================================
# 2. D3QN 模型
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
        self.value_stream = nn.Sequential(
            nn.Linear(cfg.hidden_dim_2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(cfg.hidden_dim_2, output_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


# ==========================================
# 3. One-Hot 转换（NumPy 版本）
# ==========================================
def to_onehot(state_arr, cfg):
    if len(state_arr.shape) == 1:
        state_arr = state_arr[np.newaxis, :]

    batch_size = state_arr.shape[0]
    num_classes = cfg.num_services + 1

    one_hot = np.zeros((batch_size, cfg.num_users, num_classes), dtype=np.float32)

    for i in range(batch_size):
        for u in range(cfg.num_users):
            val = state_arr[i, u]
            one_hot[i, u, int(val)] = 1.0

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
        w_item = int(sizes[i - 1])
        v_item = q_values[i - 1]
        for w in range(1, C + 1):
            if w_item <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_item] + v_item)
            else:
                dp[i][w] = dp[i - 1][w]
    selected = np.zeros(K)
    w = C
    for i in range(K, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected[i - 1] = 1
            w -= int(sizes[i - 1])
    return selected


# ==========================================
# 5. Agent
# ==========================================
class D3QNAgent:
    def __init__(self, config):
        self.cfg = config

        self.policy_net = D3QN_Network(config.feature_dim, config.num_services).to(config.device)
        self.target_net = D3QN_Network(config.feature_dim, config.num_services).to(config.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.memory = deque(maxlen=config.memory_size)
        self.loss_fn = nn.MSELoss()
        self.epsilon = config.epsilon_start
        self.steps = 0

    def select_action(self, state):
        self.steps += 1
        state_onehot = to_onehot(state, self.cfg)

        if random.random() < self.epsilon:
            random_q = np.random.rand(self.cfg.num_services)
            return solve_knapsack(random_q, self.cfg.service_sizes, self.cfg.cache_capacity)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_onehot).to(self.cfg.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                return solve_knapsack(q_values, self.cfg.service_sizes, self.cfg.cache_capacity)

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.cfg.batch_size)
        state, action, reward, next_state = zip(*batch)

        state_onehot = to_onehot(np.array(state), self.cfg)
        next_state_onehot = to_onehot(np.array(next_state), self.cfg)

        state_t = torch.FloatTensor(state_onehot).to(self.cfg.device)
        next_state_t = torch.FloatTensor(next_state_onehot).to(self.cfg.device)
        action_t = torch.FloatTensor(np.array(action)).to(self.cfg.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.cfg.device)

        all_q_values = self.policy_net(state_t)
        current_q_value = torch.sum(all_q_values * action_t, dim=1, keepdim=True)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_t)
            next_q_target = self.target_net(next_state_t)
            avg_items = max(1, int(self.cfg.cache_capacity / 2))
            _, top_k_indices = torch.topk(next_q_policy, avg_items, dim=1)
            mask = torch.zeros_like(next_q_target)
            mask.scatter_(1, top_k_indices, 1)
            max_next_q = torch.sum(next_q_target * mask, dim=1, keepdim=True)

            target_q_value = reward_t + self.cfg.gamma * max_next_q

        loss = self.loss_fn(current_q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

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
            e_local = self.cfg.zeta * (self.cfg.f_local ** 2) * S_k
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
        final_strategies = self.find_nash_equilibrium(action, next_state)

        total_energy = 0
        penalty_count = 0
        for u in range(self.cfg.num_users):
            req = next_state[u]
            if req == 0: continue
            task_id = req - 1
            s = final_strategies[u]
            energy = self._calculate_cost_for_user(u, s, final_strategies, action, task_id)
            total_energy += energy
            if energy > self.cfg.penalty / 2:  # 简单判断是否含罚分
                penalty_count += 1

        reward = -total_energy
        return next_state, reward, False, penalty_count

    def _generate_requests(self):
        # 强 Zipf，降低学习难度
        probs = 1.0 / (np.arange(1, self.cfg.num_services + 2) ** 1.5)
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

    episodes = 3000
    print(f"开始训练 | 修正版 (One-Hot + 物理可行性) | 罚分: {cfg.penalty}")

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        total_penalties = 0

        for t in range(20):
            action = agent.select_action(state)
            next_state, reward, done, pens = env.step(action)
            agent.store_experience(state, action, reward, next_state)
            loss = agent.update()
            state = next_state
            total_reward += reward
            total_penalties += pens
            if t % cfg.target_update == 0:
                agent.update_target_network()

        avg_reward = total_reward / 20

        if (e + 1) % 50 == 0:
            print(
                f"Episode {e + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Timeouts: {total_penalties}")
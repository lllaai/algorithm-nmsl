import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# ========== 设备：优先使用 MPS ==========
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
print("Using device:", DEVICE)


# ========== 按 Table I 参数的 MEC 环境 ==========
class ServiceCachingEnvTable1:
    """
    单 ES 版本，参数尽量按 Table I 设定：
      - Number of tasks: K ≈ 40
      - Number of MUs: 2000
      - ES cache size C ∈ [3, 15] GB，这里默认取中间值 10
      - CPU capability of ES: 20 GHz
      - I_max = 8, D_max = 8
      - τ = 300 ms
      - replay buffer size E = 1000（在 Agent 里用）
    """

    def __init__(
        self,
        num_users=2000,
        num_tasks=40,
        cache_capacity_gb=10.0,  # Table I: [3, 15] GB，取 10
        seed=0,
    ):
        self.U = num_users
        self.K = num_tasks
        self.C = cache_capacity_gb  # “容量单位”与 D_k 一致

        self.rng = np.random.default_rng(seed)

        # ===== 任务参数：I_k, D_k, S_k =====
        # I_max = 8, D_max = 8（Table I），我们让 I_k, D_k 在 [1, 8] 均匀分布
        self.I_k = self.rng.uniform(1.0, 8.0, size=self.K)     # 输入数据（相对单位）
        self.D_k = self.rng.uniform(1.0, 8.0, size=self.K)     # 服务大小（“GB 单位”）
        self.S_k = self.rng.uniform(10.0, 50.0, size=self.K)   # 计算量（抽象 CPU cycles）

        # ===== 计算能力 / 时隙长度 =====
        # ES CPU capability: 20 GHz
        # 这里做归一化，把 f_e / f_c 看成“单位时间内可处理的 S_k 数量”
        self.f_e = 20.0      # ES 计算能力（相对）
        self.f_c = 50.0      # 云端计算能力（比边缘更强）
        # τ = 300 ms，取 0.3 秒，再用作时延上限
        self.tau = 0.3

        # 本地执行参数：E_local = ζ * f^2 * S_k
        self.zeta = 1e-3
        self.f_local_min = 5.0

        # 无线参数（简化版）
        self.W = 5.0        # 单信道带宽
        self.noise_power = 1.0
        self.tx_power_mu = 1.0

        # ES→云链路速率（有线链路）
        self.r_ec = 20.0

        # 能耗系数（传输 / 执行）
        self.p_edge_trans = 1.0
        self.p_edge_exec = 1.0
        self.p_cloud = 1.0

        # 用户位置 / pathloss
        self.area_size = 1000.0
        self.pathloss_exp = 3.5
        self.mu_positions = self._random_mu_positions()

        # 状态：当前缓存与请求
        self.beta_t = np.zeros(self.K, dtype=int)  # 当前缓存
        self.xi_t = None                           # 当前请求（长度 U）

    # ---------- 辅助：随机用户位置 ----------
    def _random_mu_positions(self):
        xs = self.rng.uniform(0, self.area_size, size=self.U)
        ys = self.rng.uniform(0, self.area_size, size=self.U)
        return np.stack([xs, ys], axis=1)

    def _distance_to_bs(self, u):
        bs = np.array([self.area_size / 2, self.area_size / 2])
        return np.linalg.norm(self.mu_positions[u] - bs) + 1e-3

    def _uplink_rate(self, u, n_users_on_same_channel=1):
        d_u = self._distance_to_bs(u)
        g_u = d_u ** (-self.pathloss_exp)
        interference = (n_users_on_same_channel - 1) * self.tx_power_mu * g_u
        sinr = self.tx_power_mu * g_u / (self.noise_power + interference)
        rate = self.W * np.log2(1.0 + sinr)
        return max(rate, 1e-3)

    # ---------- MDP：reset / 状态表示 / step ----------
    def reset(self):
        self.beta_t[:] = 0
        self.mu_positions = self._random_mu_positions()
        self.xi_t = self._generate_requests()
        return self._state_representation()

    def _generate_requests(self):
        # 论文中 ξ^t 用随机函数模拟，我们这里：每个 MU 以 0.7 概率请求一类服务
        p_req = 0.7
        xi = []
        for _ in range(self.U):
            if self.rng.random() < p_req:
                task_k = self.rng.integers(1, self.K + 1)
                xi.append(task_k)
            else:
                xi.append(0)
        return np.array(xi, dtype=int)

    def _state_representation(self):
        # 和你之前一样，用“任务请求直方图 / U”作为状态（K 维向量）
        hist = np.zeros(self.K, dtype=float)
        for v in self.xi_t:
            if v != 0:
                hist[v - 1] += 1
        state = hist / float(self.U)
        return state.astype(np.float32)

    def step(self, beta_next):
        beta_next = np.array(beta_next, dtype=int)

        # 生成下一时隙请求
        xi_next = self._generate_requests()

        total_energy = 0.0

        # 简化卸载决策：每个 MU 在本地 / 边缘 / 云中选能耗最小且满足时延约束的一种
        for u in range(self.U):
            req = xi_next[u]
            if req == 0:
                continue
            k = req - 1

            I_k = self.I_k[k]
            D_k = self.D_k[k]
            S_k = self.S_k[k]

            # 1) 本地执行
            f_local = max(self.f_local_min, S_k / self.tau)
            E_local = self.zeta * (f_local ** 2) * S_k
            D_local = S_k / f_local
            if D_local > self.tau:
                E_local = float("inf")

            # 2) 边缘执行（要求服务已缓存）
            r_ue = self._uplink_rate(u, n_users_on_same_channel=1)
            D_edge = S_k / self.f_e + I_k / r_ue
            if D_edge <= self.tau and beta_next[k] == 1:
                E_edge = self.p_edge_trans * I_k / r_ue + self.p_edge_exec * S_k / self.f_e
            else:
                E_edge = float("inf")

            # 3) 云执行
            D_cloud = S_k / self.f_c + I_k / r_ue + I_k / self.r_ec
            if D_cloud <= self.tau:
                E_cloud = self.p_cloud * (I_k / r_ue + I_k / self.r_ec)
            else:
                E_cloud = float("inf")

            E_candidates = [E_local, E_edge, E_cloud]
            E_u_best = min(E_candidates)
            if not np.isfinite(E_u_best):
                E_u_best = 1e3  # 作为惩罚

            total_energy += E_u_best

        reward = -total_energy   # 论文目标：最小化总能耗

        self.beta_t = beta_next
        self.xi_t = xi_next
        next_state = self._state_representation()
        done = False
        return next_state, reward, done, {}


# ========== D3QN 网络（保持你原来的 dueling + per-task Q） ==========
class D3QNNet(nn.Module):
    def __init__(self, state_dim, task_count, hidden1=300, hidden2=400):
        super(D3QNNet, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.value_layer = nn.Linear(hidden2, 1)
        self.adv_layer = nn.Linear(hidden2, task_count)

    def forward(self, state):
        x = self.feature_layer(state)
        value = self.value_layer(x)       # (B,1)
        advantage = self.adv_layer(x)     # (B,K)
        adv_mean = advantage.mean(dim=1, keepdim=True)
        q_part = value + (advantage - adv_mean)
        return q_part                     # (B,K)


# ========== 简单 Replay Buffer（大小 = Table I 的 1000） ==========
class ReplayMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, ns, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ========== D3QN Agent（参数参考论文：lr=1e-3, gamma=0.95, buffer=1000） ==========
class D3QNAgent:
    def __init__(
        self,
        state_dim,
        task_count,
        cache_sizes,
        cache_capacity,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        target_tau=0.01,
        device=DEVICE,
    ):
        self.state_dim = state_dim
        self.K = task_count
        self.cache_sizes = np.array(cache_sizes, dtype=float)
        self.capacity = float(cache_capacity)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_tau = target_tau
        self.device = device

        self.eval_net = D3QNNet(state_dim, task_count).to(device)
        self.target_net = D3QNNet(state_dim, task_count).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=1000)
        self.learn_step = 0

    def select_action(self, state, greedy=False):
        if isinstance(state, np.ndarray):
            s_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        else:
            s_tensor = state.unsqueeze(0).float().to(self.device)

        eps = 0.0 if greedy else self.epsilon

        if random.random() > eps:
            with torch.no_grad():
                q_parts = self.eval_net(s_tensor)  # (1,K)
                q_np = q_parts.cpu().numpy().squeeze()
            action_vec = self._solve_knapsack(q_np)
        else:
            action_vec = self._random_action()

        if (not greedy) and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end

        return action_vec

    def _random_action(self):
        idx = list(range(self.K))
        random.shuffle(idx)
        total = 0.0
        a = [0] * self.K
        for k in idx:
            size = self.cache_sizes[k]
            if total + size <= self.capacity:
                if random.random() < 0.5:
                    a[k] = 1
                    total += size
        return np.array(a, dtype=int)

    def _solve_knapsack(self, values):
        K = self.K
        # 用整数近似容量和大小（D_k 在 [1,8]，C=10，下界不高）
        weights = self.cache_sizes.astype(int)
        C = int(self.capacity)

        dp = [0.0] * (C + 1)
        choose = [[0] * (C + 1) for _ in range(K + 1)]

        for i in range(1, K + 1):
            w = weights[i - 1]
            v = float(values[i - 1])
            for cap in range(C, -1, -1):
                if w <= cap:
                    new_v = dp[cap - w] + v
                    if new_v > dp[cap]:
                        dp[cap] = new_v
                        choose[i][cap] = 1

        beta = [0] * K
        cap = C
        for i in range(K, 0, -1):
            if choose[i][cap] == 1:
                beta[i - 1] = 1
                cap -= weights[i - 1]
        return np.array(beta, dtype=int)

    def store_transition(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)

        states = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        actions = np.stack([b[1] for b in batch], axis=0).astype(np.float32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch], axis=0).astype(np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        s_batch = torch.from_numpy(states).to(self.device)
        a_batch = torch.from_numpy(actions).to(self.device)
        r_batch = torch.from_numpy(rewards).to(self.device)
        ns_batch = torch.from_numpy(next_states).to(self.device)
        d_batch = torch.from_numpy(dones).to(self.device)

        q_parts = self.eval_net(s_batch)                      # (B,K)
        q_sa = torch.sum(q_parts * a_batch, dim=1)            # (B,)

        with torch.no_grad():
            next_q_eval = self.eval_net(ns_batch)             # (B,K)
            next_actions = []
            for i in range(next_q_eval.size(0)):
                q_np = next_q_eval[i].cpu().numpy()
                beta_best = self._solve_knapsack(q_np)
                next_actions.append(beta_best)
            next_actions = np.stack(next_actions, axis=0).astype(np.float32)
            next_actions_t = torch.from_numpy(next_actions).to(self.device)

            next_q_target = self.target_net(ns_batch)         # (B,K)
            q_next = torch.sum(next_q_target * next_actions_t, dim=1)

            target = r_batch + self.gamma * q_next * (1.0 - d_batch)

        loss = nn.MSELoss()(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update
        with torch.no_grad():
            for tp, ep in zip(self.target_net.parameters(), self.eval_net.parameters()):
                tp.data.copy_(self.target_tau * ep.data + (1.0 - self.target_tau) * tp.data)

        self.learn_step += 1
        return loss.item()


# ========== 训练主程序 ==========
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = ServiceCachingEnvTable1(
        num_users=2000,
        num_tasks=40,
        cache_capacity_gb=10.0,
        seed=0,
    )

    agent = D3QNAgent(
        state_dim=env.K,
        task_count=env.K,
        cache_sizes=env.D_k,
        cache_capacity=env.C,
        device=DEVICE,
    )

    num_episodes = 800      # 2000 MU * 40 任务，单次 episode 已经很重了，先跑 800 看趋势
    max_steps = 30          # 每个 episode 内的时隙数
    batch_size = 128

    returns = []

    for ep in range(num_episodes):
        state = env.reset()
        episode_return = 0.0

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update(batch_size)

            episode_return += reward
            state = next_state
            if done:
                break

        returns.append(episode_return)

        if (ep + 1) % 20 == 0:
            recent = returns[-100:] if len(returns) >= 100 else returns
            avg_recent = sum(recent) / len(recent)
            print(
                f"Episode {ep+1}: Return = {episode_return:.2f}, "
                f"Avg(Last {len(recent)}) = {avg_recent:.2f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

    print("训练完成。")
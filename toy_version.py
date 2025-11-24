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


# ========== Toy 环境：只考虑“缓存命中次数” ==========
class ToyCacheEnv:
    """
    非 MEC，只是验证 D3QN+背包 能否学会“缓存热门任务”。

    设：
      - K = 5 个任务
      - 容量 C = 2，只能缓存两个任务
      - 每步有 U = 20 个请求，每个请求按固定热门度向量 p 采样任务
      - 动作：beta ∈ {0,1}^K，且 sum(beta) <= 2
      - reward = 本步缓存命中次数（命中越多越好）
      - 状态：上一时刻请求直方图 / U（反映热门度，但有随机噪声）
    """

    def __init__(self, num_tasks=5, users_per_step=20, seed=0):
        self.K = num_tasks
        self.U = users_per_step
        self.rng = np.random.default_rng(seed)

        # 任务热门度（固定不变）：task 0、1 非常热门
        self.popularity = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        self.popularity /= self.popularity.sum()

        # 容量：最多缓存 2 个任务
        self.C = 2

        # 当前请求直方图 state
        self.last_hist = None

    def reset(self):
        # 初始化：先模拟一轮请求得到初始 state
        hist = self._simulate_requests()
        self.last_hist = hist
        return hist.astype(np.float32)

    def _simulate_requests(self):
        """
        模拟 U 个用户请求，返回 K 维直方图（比例）。
        """
        # 任务索引 0..K-1
        reqs = self.rng.choice(self.K, size=self.U, p=self.popularity)
        hist = np.zeros(self.K, dtype=float)
        for k in reqs:
            hist[k] += 1.0
        hist = hist / float(self.U)
        return hist

    def step(self, beta):
        """
        beta: 长度 K 的 0/1 向量，表示缓存哪些任务（最多缓存 2 个）
        """
        beta = np.array(beta, dtype=int)

        # 容量约束：如果超过 2 个，就只保留值最大的两个 1
        if beta.sum() > self.C:
            idx = np.where(beta == 1)[0]
            # 随机保留两个，也可以按任务 index 排序保留前两个
            self.rng.shuffle(idx)
            keep = idx[: self.C]
            new_beta = np.zeros_like(beta)
            new_beta[keep] = 1
            beta = new_beta

        # 模拟本步请求
        hist = self._simulate_requests()  # 本步请求直方图（比例）
        # 转换为次数
        counts = hist * self.U

        # 命中次数 = sum_k beta_k * counts_k
        hits = float(np.dot(beta, counts))

        reward = hits  # 正向奖励：命中越多越好

        # 下一个状态就是当前这步的直方图
        next_state = hist.astype(np.float32)
        self.last_hist = next_state
        done = False

        return next_state, reward, done, {}


# ========== D3QN 网络：和前面类似，但规模小 ==========
class D3QNNet(nn.Module):
    def __init__(self, state_dim, task_count, hidden_size=64):
        super(D3QNNet, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_layer = nn.Linear(hidden_size, 1)
        self.adv_layer = nn.Linear(hidden_size, task_count)

    def forward(self, state):
        x = self.feature_layer(state)
        value = self.value_layer(x)        # (B,1)
        adv = self.adv_layer(x)            # (B,K)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_part = value + (adv - adv_mean)  # (B,K)
        return q_part


# ========== Replay Buffer ==========
class ReplayMemory:
    def __init__(self, capacity=5000):
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


# ========== D3QN Agent（动作仍然是“背包选择任务”）==========
class ToyD3QNAgent:
    def __init__(
        self,
        state_dim,
        task_count,
        cache_capacity,     # 这里就是最多缓存几个任务（2）
        lr=1e-3,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_tau=0.01,
        device=DEVICE,
    ):
        self.state_dim = state_dim
        self.K = task_count
        self.capacity = int(cache_capacity)  # 容量 = 能缓存任务个数（每个任务大小视为1）

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
        self.memory = ReplayMemory(capacity=5000)

    def select_action(self, state, greedy=False):
        if isinstance(state, np.ndarray):
            s_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        else:
            s_tensor = state.unsqueeze(0).float().to(self.device)

        eps = 0.0 if greedy else self.epsilon

        if random.random() > eps:
            with torch.no_grad():
                q_parts = self.eval_net(s_tensor)   # (1,K)
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
        # 随机选择 capacity 个任务缓存
        idx = list(range(self.K))
        random.shuffle(idx)
        chosen = idx[: self.capacity]
        a = np.zeros(self.K, dtype=int)
        a[chosen] = 1
        return a

    def _solve_knapsack(self, values):
        """
        因为每个任务“大小=1”，容量=capacity，这里其实就是选 values 最大的 capacity 个。
        但为了跟之前代码风格一致，仍叫 _solve_knapsack。
        """
        idx_sorted = np.argsort(values)[::-1]  # 从大到小排序
        chosen = idx_sorted[: self.capacity]
        beta = np.zeros(self.K, dtype=int)
        beta[chosen] = 1
        return beta

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

        # 当前 Q(s,a) = sum_k a_k * q_part_k
        q_parts = self.eval_net(s_batch)                 # (B,K)
        q_sa = torch.sum(q_parts * a_batch, dim=1)       # (B,)

        with torch.no_grad():
            next_q_eval = self.eval_net(ns_batch)        # (B,K)
            next_actions = []
            for i in range(next_q_eval.size(0)):
                q_np = next_q_eval[i].cpu().numpy()
                beta_best = self._solve_knapsack(q_np)
                next_actions.append(beta_best)
            next_actions = np.stack(next_actions, axis=0).astype(np.float32)
            next_actions_t = torch.from_numpy(next_actions).to(self.device)

            next_q_target = self.target_net(ns_batch)    # (B,K)
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

        return loss.item()

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = ToyCacheEnv(num_tasks=5, users_per_step=20, seed=0)
    agent = ToyD3QNAgent(
        state_dim=env.K,
        task_count=env.K,
        cache_capacity=2,   # 容量：最多缓存两个任务
        device=DEVICE,
    )

    num_episodes = 300
    max_steps = 20
    batch_size = 64
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

        returns.append(episode_return)

        if (ep + 1) % 20 == 0:
            recent = returns[-50:] if len(returns) >= 50 else returns
            avg_recent = sum(recent) / len(recent)
            print(
                f"Episode {ep+1}: Return = {episode_return:.2f}, "
                f"Avg(Last {len(recent)}) = {avg_recent:.2f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

    print("Toy 环境训练完成。")

    # ========== 训练结束后，看学到的策略 vs 理论最优 ==========
    # 理论最优：缓存 popularity 最大的两个任务
    true_pop = env.popularity
    opt_idx = np.argsort(true_pop)[::-1][:2]
    opt_beta = np.zeros(env.K, dtype=int)
    opt_beta[opt_idx] = 1

    # 用训练好的 agent 在“平均状态”（popularity 本身）下选动作
    avg_state = true_pop.astype(np.float32)  # 理论上状态直方图的期望就是 popularity
    learned_beta = agent.select_action(avg_state, greedy=True)

    print("\n理论热门度向量 p:", true_pop)
    print("理论最优缓存策略 beta*:", opt_beta, "（缓存任务索引:", opt_idx, ")")
    print("Agent 学到的贪心缓存策略:", learned_beta)
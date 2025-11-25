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


# ========== Basic MEC v3 环境 ==========
class BasicMECEnvV3:
    """
    目标：MEC 风格 + 强可学习信号。

    - U = 50 个用户
    - K = 10 个任务
    - 每 step：每个用户必定请求一个任务，按固定热门度分布 p 采样
    - 动作 beta ∈ {0,1}^K：缓存哪些任务（容量 = 可缓存 3 个任务）
    - 能耗模型（极简但有 MEC 意义）：
        * 视角：用户要么本地执行，要么卸载到 ES
        * 未缓存：等价于“本地执行”或“云执行”，统一视为很贵：E_miss = 10
        * 缓存命中：卸载到边缘执行，很省：E_hit = 1
      所以每个请求的能耗 E = 10 - 9 * 1{beta_k=1}
    - 总能耗：E_total = 10 * U - 9 * hits
    - reward = - E_total / U = -10 + 9 * (hits / U)
      ∈ [-10, -1]，越大越好（命中率越高）
    - 状态：当前时隙请求直方图 / U （K 维）
    """

    def __init__(self, num_users=50, num_tasks=10, seed=0):
        self.U = num_users
        self.K = num_tasks
        self.rng = np.random.default_rng(seed)

        # 任务热门度：Zipf 分布，前几个任务非常热门
        ranks = np.arange(1, self.K + 1, dtype=float)
        alpha = 1.0
        raw_pop = 1.0 / (ranks ** alpha)
        self.popularity = raw_pop / raw_pop.sum()  # p_k

        # 缓存容量：最多缓存 3 个任务（每个任务大小=1）
        self.C = 3

        # 当前请求（长度 U，每个是 0..K-1）
        self.current_requests = None

    def reset(self):
        self.current_requests = self._generate_requests()
        state = self._state_from_requests(self.current_requests)
        return state.astype(np.float32)

    def _generate_requests(self):
        # 每个用户必定请求一个任务，按热门度分布采样
        reqs = self.rng.choice(self.K, size=self.U, p=self.popularity)
        return reqs

    def _state_from_requests(self, reqs):
        hist = np.zeros(self.K, dtype=float)
        for k in reqs:
            hist[k] += 1.0
        hist /= float(self.U)
        return hist

    def step(self, beta):
        """
        beta: 长度 K 的 0/1 向量，表示缓存哪些任务。sum(beta) <= C。
        """
        beta = np.array(beta, dtype=int)

        # 容量约束：若超过 C，只保留任意 C 个 1（简单处理）
        if beta.sum() > self.C:
            idx = np.where(beta == 1)[0]
            self.rng.shuffle(idx)
            keep = idx[: self.C]
            new_beta = np.zeros_like(beta)
            new_beta[keep] = 1
            beta = new_beta

        # 计算本步命中次数
        hits = 0
        for u in range(self.U):
            k = self.current_requests[u]
            if beta[k] == 1:
                hits += 1
        hits = float(hits)

        # 总能耗 & reward
        # E_total = 10*U - 9*hits
        E_total = 10.0 * self.U - 9.0 * hits
        reward = -E_total / self.U  # ∈ [-10, -1]，越大越好

        # 下一时隙请求 & 状态
        self.current_requests = self._generate_requests()
        next_state = self._state_from_requests(self.current_requests).astype(np.float32)
        done = False
        return next_state, reward, done, {}


# ========== D3QN 网络 ==========
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
        value = self.value_layer(x)
        adv = self.adv_layer(x)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_part = value + (adv - adv_mean)
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


# ========== Agent：动作仍然是“选 C 个任务缓存” ==========
class BasicMECv3Agent:
    def __init__(
            self,
            state_dim,
            task_count,
            cache_capacity,  # 这里是“可缓存任务个数 C”
            lr=1e-3,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99,
            target_tau=0.01,
            device=DEVICE,
    ):
        self.state_dim = state_dim
        self.K = task_count
        self.capacity = int(cache_capacity)

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
                q_parts = self.eval_net(s_tensor)  # (1,K)
                q_np = q_parts.cpu().numpy().squeeze()
            action_vec = self._select_top_k(q_np)
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
        chosen = idx[: self.capacity]
        a = np.zeros(self.K, dtype=int)
        a[chosen] = 1
        return a

    def _select_top_k(self, values):
        idx_sorted = np.argsort(values)[::-1]
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

        q_parts = self.eval_net(s_batch)
        q_sa = torch.sum(q_parts * a_batch, dim=1)

        with torch.no_grad():
            next_q_eval = self.eval_net(ns_batch)
            next_actions = []
            for i in range(next_q_eval.size(0)):
                q_np = next_q_eval[i].cpu().numpy()
                beta_best = self._select_top_k(q_np)
                next_actions.append(beta_best)
            next_actions = np.stack(next_actions, axis=0).astype(np.float32)
            next_actions_t = torch.from_numpy(next_actions).to(self.device)

            next_q_target = self.target_net(ns_batch)
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


# ========== 训练主程序 ==========
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = BasicMECEnvV3(num_users=50, num_tasks=10, seed=0)
    agent = BasicMECv3Agent(
        state_dim=env.K,
        task_count=env.K,
        cache_capacity=env.C,  # = 3
        device=DEVICE,
    )

    num_episodes = 300
    max_steps = 10
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
                f"Episode {ep + 1}: Return = {episode_return:.2f}, "
                f"Avg(Last {len(recent)}) = {avg_recent:.2f}, "
                f"epsilon={agent.epsilon:.3f}"
            )

    print("Basic MEC v3 训练完成。")

    # 看最终策略缓存了哪些任务
    avg_state = env.popularity.astype(np.float32)
    greedy_action = agent.select_action(avg_state, greedy=True)
    print("任务热门度 p_k:", env.popularity)
    print("Agent 贪心缓存策略 beta*:", greedy_action)
    print("缓存任务索引:", np.where(greedy_action == 1)[0])
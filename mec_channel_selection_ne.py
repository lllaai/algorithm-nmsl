import math
import random
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple

# =========================
# 基本数据结构
# =========================

@dataclass
class Task:
    """
    对应论文中的任务 k ∈ K:
    (Dk, Ik, Sk)
    Dk: 对应服务占用的缓存大小 (GB 或 MB)
    Ik: 任务输入数据大小 (bit 或 Byte, 自己统一单位)
    Sk: 所需 CPU 周期数
    """
    D: float   # service size
    I: float   # input size
    S: float   # required CPU cycles


@dataclass
class User:
    """
    用户 u ∈ U 的参数:
    q_tx: 传输功率 q_u
    g: 与 ES 之间的信道增益 g_u^e
    zeta: 本地计算能耗系数 ζ
    tau: 时间片长度 τ
    task_id: 当前要执行的任务编号 (k)，若无任务可设为 -1
    """
    q_tx: float
    g: float
    zeta: float
    tau: float
    task_id: int


@dataclass
class SystemParams:
    """
    系统级参数:
    W: 总带宽 (Hz)
    N: 信道数 (编号 1..N; 0 表示本地执行)
    noise_power: 噪声功率 δ^2
    r_es_cloud: ES->Cloud 的固定传输速率 r_e,c
    f_edge: 边缘服务器计算能力 f_e (cycles/s)
    f_cloud: 云端计算能力 f_c (cycles/s)
    p_edge_trans: 边缘卸载时的传输功率 p_k,trans (近似可视为用户功率)
    p_edge_exe: ES 执行任务时的功率 p_k,exe
    p_cloud: 卸载到云端时的传输功率 p_k (与用户上行相同也可以)
    beta: 当前时间片 ES 的服务缓存向量 β^t (长度 K，0/1)
    """
    W: float
    N: int
    noise_power: float
    r_es_cloud: float
    f_edge: float
    f_cloud: float
    p_edge_trans: float
    p_edge_exe: float
    p_cloud: float
    beta: List[int]


# =========================
# 多用户信道选择算法 (Algorithm 1)
# =========================

Eta = List[int]  # 长度为 U，eta[u] 表示用户 u 的信道选择 (0=local, 1..N)

CostFunction = Callable[[Eta], float]
StrategySpaceFunction = Callable[[int, Eta], List[int]]


class MultiUserChannelSelector:
    """
    对应论文中的 Algorithm 1: Multiuser CSD Based on β*_t+1
    """

    def __init__(
        self,
        num_users: int,
        num_channels: int,
        cost_fn: CostFunction,
        strategy_space_fn: StrategySpaceFunction,
        max_iterations: int = 1000,
        verbose: bool = False,
    ):
        self.U = num_users
        self.N = num_channels
        self.cost_fn = cost_fn
        self.strategy_space_fn = strategy_space_fn
        self.max_iterations = max_iterations
        self.verbose = verbose

    def best_response(self, u: int, eta: Eta) -> int:
        """
        对用户 u 求 Best Response:
        η*_u = argmin_{a ∈ Ω_u} cost(η_{-u}, a)
        """
        current_action = eta[u]
        current_cost = self.cost_fn(eta)

        candidates = self.strategy_space_fn(u, eta)
        if not candidates:
            return current_action

        best_action = current_action
        best_cost = current_cost

        for a in candidates:
            if a == current_action:
                continue
            tmp_eta = eta.copy()
            tmp_eta[u] = a
            c = self.cost_fn(tmp_eta)
            if c < best_cost:
                best_cost = c
                best_action = a

        return best_action

    def run(self, initial_eta: Optional[Eta] = None) -> Tuple[Eta, float, int]:
        """
        返回:
        - eta_star: 近似纳什均衡
        - final_cost: 对应系统成本
        - iters: 实际迭代轮数
        """
        if initial_eta is None:
            eta = [0 for _ in range(self.U)]  # 全部本地执行
        else:
            eta = initial_eta.copy()

        if self.verbose:
            print(f"[INIT] eta = {eta}, cost = {self.cost_fn(eta):.4f}")

        for it in range(1, self.max_iterations + 1):
            update_requests = []

            # 每个用户算一遍自己的 best response
            for u in range(self.U):
                br = self.best_response(u, eta)
                if br != eta[u]:
                    update_requests.append((u, br))

            if not update_requests:
                if self.verbose:
                    print(f"[END] No more updates at iter {it}.")
                break

            # ES 随机批准一个用户更新（论文中的“随机授权”）
            u_sel, a_sel = random.choice(update_requests)
            if self.verbose:
                print(f"[ITER {it}] User {u_sel}: {eta[u_sel]} -> {a_sel}")
            eta[u_sel] = a_sel

        final_cost = self.cost_fn(eta)
        if self.verbose:
            print(f"[RESULT] eta* = {eta}, cost = {final_cost:.4f}")

        return eta, final_cost, it


# =========================
# 能量模型实现（对应论文公式）
# =========================

def uplink_rate(u_idx: int, eta: Eta, users: List[User], params: SystemParams) -> float:
    """
    对应 (2) 的上行速率 r_u,e^t:
    r = W_n * log2(1 + q_u g_u / (noise + sum_{v≠u, same channel} q_v g_v))

    这里简单假设:
    - 总带宽 W 被 N 个信道平均分配: W_n = W / N
    - eta[u] == 0 表示本地执行，此时速率无意义，返回 0
    """
    ch = eta[u_idx]
    if ch == 0:
        return 0.0

    W_n = params.W / params.N
    q_u = users[u_idx].q_tx
    g_u = users[u_idx].g

    interference = 0.0
    for v_idx, ch_v in enumerate(eta):
        if v_idx == u_idx:
            continue
        if ch_v == ch:  # 同一信道上的干扰
            interference += users[v_idx].q_tx * users[v_idx].g

    snr = q_u * g_u / (params.noise_power + interference)
    if snr <= 0:
        return 0.0

    r = W_n * math.log2(1.0 + snr)
    return r


def energy_local(user: User, task: Task) -> float:
    """
    对应公式 (3) 本地计算能耗:
    E_l = ζ * f_l,k^2 * S_k
    且 f_l,k >= S_k / τ
    这里取刚好 f_l,k = S_k / τ
    """
    if user.task_id < 0:
        return 0.0
    f_local = task.S / user.tau
    return user.zeta * (f_local ** 2) * task.S


def energy_cloud(u_idx: int, eta: Eta, users: List[User], task: Task, params: SystemParams) -> float:
    """
    对应公式 (5) 卸载到云端的能耗:
    E_c = p_k * (I_k / r_u,e + I_k / r_e,c)
    这里用 params.p_cloud 作为 p_k
    """
    r_ue = uplink_rate(u_idx, eta, users, params)
    if r_ue <= 0:
        # 速率过低视为不可用，这里简单给一个极大值
        return 1e12

    t_upl = task.I / r_ue
    t_es_cloud = task.I / params.r_es_cloud
    return params.p_cloud * (t_upl + t_es_cloud)


def energy_edge(u_idx: int, eta: Eta, users: List[User], task: Task, params: SystemParams) -> float:
    """
    对应公式 (7) 卸载到 ES 的能耗:
    E_e = p_k,trans * I_k / r_u,e + p_k,exe * S_k / f_e
    """
    r_ue = uplink_rate(u_idx, eta, users, params)
    if r_ue <= 0:
        return 1e12

    t_upl = task.I / r_ue
    t_exec = task.S / params.f_edge
    return params.p_edge_trans * t_upl + params.p_edge_exe * t_exec


def system_cost(eta: Eta, users: List[User], tasks: List[Task], params: SystemParams) -> float:
    """
    对应公式 (8) 系统总能耗:
    对每个用户 u，根据 eta[u] 和 β^t_k 决定:
    - eta[u] = 0 → 本地执行 → E_l
    - eta[u] != 0 & β_k = 1 → ES 执行 → E_e
    - eta[u] != 0 & β_k = 0 → 云端执行 → E_c
    """
    total = 0.0
    for u_idx, user in enumerate(users):
        k = user.task_id
        if k < 0:
            continue  # 没有任务

        task = tasks[k]

        if eta[u_idx] == 0:
            total += energy_local(user, task)
        else:
            if params.beta[k] == 1:
                total += energy_edge(u_idx, eta, users, task, params)
            else:
                total += energy_cloud(u_idx, eta, users, task, params)
    return total


# =========================
# 策略空间 (简化版的“可行信道集合”)
# =========================

def strategy_space(
    u_idx: int,
    eta: Eta,
    users: List[User],
    tasks: List[Task],
    params: SystemParams,
) -> List[int]:
    """
    近似实现论文中 “只在卸载比本地更划算时才考虑信道”的思想
    对应 (32) 的精神：
        (1 - β_k) E_c + β_k E_e <= E_l

    做法：
    - 所有信道 {0..N} 都先看一遍
    - 对 a == 0: 总是允许 (本地执行)
    - 对 a > 0:
        临时设 eta[u] = a，计算本地能耗 E_l 和 卸载能耗 E_off
        若 E_off < E_l，则保留该信道为可行
    """
    user = users[u_idx]
    k = user.task_id
    if k < 0:
        # 无任务，随便选，这里默认允许所有（你也可以直接返回 [0]）
        return list(range(0, params.N + 1))

    task = tasks[k]
    E_l = energy_local(user, task)

    candidates = [0]  # 本地永远可行
    for ch in range(1, params.N + 1):
        tmp_eta = eta.copy()
        tmp_eta[u_idx] = ch

        if params.beta[k] == 1:
            E_off = energy_edge(u_idx, tmp_eta, users, task, params)
        else:
            E_off = energy_cloud(u_idx, tmp_eta, users, task, params)

        if E_off < E_l:
            candidates.append(ch)

    return candidates


# =========================
# 一个可直接运行的小示例
# =========================

if __name__ == "__main__":
    random.seed(0)

    # ----- 构造一些假数据 -----
    U = 5      # 用户数
    K = 3      # 任务数
    N = 3      # 信道数

    # 三个任务，随便设一些规模
    tasks = [
        Task(D=1.0, I=1e6,  S=1e9),   # 任务 0
        Task(D=1.5, I=2e6,  S=2e9),   # 任务 1
        Task(D=2.0, I=0.5e6, S=0.5e9) # 任务 2
    ]

    # 五个用户，每人随机分配一个任务
    users: List[User] = []
    for u in range(U):
        task_id = random.randrange(0, K)
        users.append(
            User(
                q_tx=0.1,          # 0.1 W
                g=1e-3 + 1e-3*u,   # 随便设一个递增的信道增益
                zeta=1e-28,        # 本地能耗系数 (类似移动芯片常用量级)
                tau=0.1,           # 时间片 0.1 s
                task_id=task_id,
            )
        )

    # 假设当前 ES 只缓存了任务 0 和 2
    beta = [1, 0, 1]

    params = SystemParams(
        W=1e6,              # 1 MHz 总带宽
        N=N,
        noise_power=1e-9,   # 噪声功率
        r_es_cloud=1e7,     # ES->Cloud 速率 10 Mbps
        f_edge=1e10,        # ES 计算能力
        f_cloud=1e11,       # Cloud 计算能力
        p_edge_trans=0.1,
        p_edge_exe=10.0,
        p_cloud=0.2,
        beta=beta,
    )

    # 定义系统成本函数
    def cost_fn(eta: Eta) -> float:
        return system_cost(eta, users, tasks, params)

    # 定义策略空间函数
    def strategy_space_fn(u_idx: int, eta: Eta) -> List[int]:
        return strategy_space(u_idx, eta, users, tasks, params)

    selector = MultiUserChannelSelector(
        num_users=U,
        num_channels=N,
        cost_fn=cost_fn,
        strategy_space_fn=strategy_space_fn,
        max_iterations=200,
        verbose=True,
    )

    eta_star, final_cost, iters = selector.run()

    print("\n=== Final Result ===")
    print("eta* =", eta_star)
    print("final cost =", final_cost)
    print("iterations =", iters)
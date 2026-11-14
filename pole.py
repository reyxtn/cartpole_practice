# 01 导入优化器模块（PPO 里要用）
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 02 导入网络激活等函数（PPO 构造常用）
import torch.nn.functional as F

# 03 导入 PPO 算法
from mushroom_rl.algorithms.actor_critic import PPO
# 04 1.10 没有内置 networks，本地定义同名类（后面代码不用改）
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input, n_output = input_shape[-1], output_shape[0]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

    def forward(self, state, **kwargs):
        x = F.relu(self._h1(torch.squeeze(state, 1).float()))
        x = F.relu(self._h2(x))
        return self._h3(x)


FeedForwardNetwork = ActorNetwork  # Critic 复用同一结构
# 05 导入 Core（把 agent 和环境串起来训练/评估）
from mushroom_rl.core import Core
# 06 导入 Gym 环境封装（1.10 里叫 Gym）
from mushroom_rl.environments import Gym as Gymnasium
# 07 导入 Boltzmann 策略（离散动作用）
from mushroom_rl.policy import BoltzmannTorchPolicy
# 08 导入可调参数 Parameter（如 beta）
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_episodes_length

# 09 创建 CartPole-v1：最长 500 步，折扣 0.99
mdp = Gymnasium("CartPole-v1", horizon=500, gamma=0.99)
# 10 写观测空间形状（这里要你填） info.observation_space：观测空间 .shape：观测向量形状，这里是 (4,)
obs_shape = mdp.info.observation_space.shape

# 11 策略 pi：当前“怎么推车”的行为模块（actor）
#    ——它本身不会自己学；后面 PPO 根据试玩数据改它网络里的权重，策略才变强
#    训练时每一步：环境给状态 s → pi 按分布抽动作 a → 环境回报 r → PPO 用这批 (s,a,r) 更新 pi
pi = BoltzmannTorchPolicy(
    ActorNetwork,                 # 网络类型：把状态映射成每个动作的分数（logits）
    obs_shape,                    # 输入维：观测形状，CartPole 是 (4,)
    (mdp.info.action_space.n,),   # 输出维：动作个数；CartPole 左右推 → n=2 → (2,)
    beta=Parameter(1.0),          # Boltzmann 温度相关：分数→概率；越大越偏向高分动作
    n_features=64,                # Actor 隐层宽度；和后面 critic 常取一样，方便训练
    # use_cuda=True 在 1.10 的 BoltzmannTorchPolicy 会报错（库里对 CUDA tensor 调了 .numpy()）
)
#    选动作逻辑（前向）：logits = Actor(s) → P(a|s) ∝ exp(beta * logits) → 按 P 采样左/右
#    更新逻辑（训练）：PPO 在 core.learn 里用优势函数改 Actor 权重，使高回报动作概率上升（带 clip）


# 12 构造 PPO agent
#    Critic：估状态价值 V(s)，给优势函数用；Actor 已在上面的 pi 里
critic_params = dict(
    network=FeedForwardNetwork,  # 价值网络结构：s → V(s)
    optimizer={                  # Critic 用 Adam 更新权重
        "class": optim.Adam,
        "params": {"lr": 3e-4},  # Critic 学习率
    },
    loss=F.mse_loss,             # 价值拟合用均方误差
    n_features=64,               # Critic 隐层宽度（与 Actor 一致）
    batch_size=64,               # Critic 每次更新用多少条样本
    input_shape=obs_shape,       # 输入维：观测形状 (4,)
    output_shape=(1,),           # 输出维：标量价值 V(s)
)

agent = PPO(
    mdp.info,                    # 环境信息：状态/动作维、折扣 γ 等
    pi,                          # 策略（Actor）：按分布选左/右
    actor_optimizer={            # Actor 用 Adam 更新权重
        "class": optim.Adam,
        "params": {"lr": 3e-4},  # Actor 学习率
    },
    critic_params=critic_params, # 上面定义的 Critic 网络与训练设置
    n_epochs_policy=4,           # 同一批轨迹对策略重复更新几遍
    batch_size=64,               # 策略更新的小批量大小
    eps_ppo=0.2,                 # PPO clip 范围 ε：限制策略一步改多大
    lam=0.95,                    # GAE 的 λ：优势估计的偏差-方差折中
)

# 13 用 agent + mdp 组成 Core
core = Core(agent, mdp)

# 14 训练循环：共 30 轮
for epoch in range(30):
    # 15 每轮学 2048 步（每 2048 步做一次更新）
    core.learn(n_steps=2048, n_steps_per_fit=2048, quiet=True)
    # 16 评估 5 局并渲染
    dataset = core.evaluate(n_episodes=5, render=True, quiet=True)
    # 17 打印本轮平均回合长度
    print(
        f"epoch {epoch + 1}: mean episode length {np.mean(compute_episodes_length(dataset)):.0f}"
    )

# 18 训练结束后再渲染评估 3 局
core.evaluate(n_episodes=3, render=True)

import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.approximators.parametric.networks import (
    ActorNetwork,
    FeedForwardNetwork,
)
from mushroom_rl.core import Core
from mushroom_rl.environments import Gymnasium
from mushroom_rl.policy import BoltzmannTorchPolicy
from mushroom_rl.rl_utils.parameters import Parameter
from mushroom_rl.approximators.parametric import TorchApproximator

mdp = Gymnasium("CartPole-v1", horizon=500, gamma=0.99)
obs_shape = mdp.info.observation_space.shape

pi = BoltzmannTorchPolicy(
    ActorNetwork,
    obs_shape,
    (mdp.info.action_space.n,),
    beta=Parameter(1.0),
    n_features=64,
)

agent = PPO(
    mdp.info,
    pi,
    actor_optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
    critic_params={
        "network": FeedForwardNetwork,
        "input_shape": obs_shape,
        "output_shape": (),
        "n_features": 64,
        "optimizer": {"class": optim.Adam, "params": {"lr": 3e-4}},
        "loss": F.mse_loss,
        "batch_size": 64,
    },
    n_epochs_policy=10,
    batch_size=64,
    eps_ppo=0.2,
    lam=0.95,
)

core = Core(agent, mdp)

for epoch in range(30):
    core.learn(n_steps=2048, n_steps_per_fit=2048, quiet=True)
    dataset = core.evaluate(n_episodes=5, render=True, quiet=True)
    print(
        f"epoch {epoch + 1}: mean episode length {dataset.episodes_length.mean():.0f}"
    )

core.evaluate(n_episodes=3, render=True)

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

mdp = Gymnasium("CartPole-v1", horizon=500, gamma=0.9)
obs_shape = (4,) # carvelocity, cart position, pole angle, pole angular velocity

pi = BoltzmannTorchPolicy(
    ActorNetwork,
    obs_shape,
    (mdp.info.action_space.n,),
    beta=Parameter(1.0),
    n_features=64,
)

critic_params = dict(
    network=FeedForwardNetwork,
    optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
    loss=F.mse_loss,
    n_features=64,
    batch_size=64,
    input_shape=obs_shape,
    output_shape=(),
)

agent = PPO(
    mdp.info,
    pi,
    actor_optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
    critic_params=critic_params,
    n_epochs_policy=4,
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
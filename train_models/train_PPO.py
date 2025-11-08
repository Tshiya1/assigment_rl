import os
import gym as old_gym
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
from CustomFeatureExtractors.impala import ImpalaCNN
from Wrappers.CrafterRewardWrapper import CrafterRewardWrapper

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
    
# Output directory for logs and stats
outdir = "logdir/crafter_reward-ppo/0"
os.makedirs(outdir, exist_ok=True)

# Register environment (ignore if already registered)
try:
    register(id='CrafterNoReward-v1', entry_point=crafter.Env)
except Exception as e:
    print("Env already registered:", e)

print("✅ Crafter environment ready!")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
env = old_gym.make('CrafterNoReward-v1')
env = CrafterRewardWrapper(env)

env = crafter.Recorder(
    env,
    outdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)

env = GymV21CompatibilityV0(env=env)
env = Monitor(env)

# Initialize PPO
policy_kwargs = dict(
    features_extractor_class=ImpalaCNN,
    features_extractor_kwargs=dict(features_dim=512),  
)

model = PPO(
    policy="CnnPolicy",
    env=env,
     n_steps=2048,
    batch_size=128,         # minibatch size
    n_epochs=3,           # epochs per rollout
    gamma=0.99,           # discount factor
    gae_lambda=0.9,      # GAE smoothing
    ent_coef=0.03,        # entropy bonus
    clip_range=0.2,       # PPO clipping
    normalize_advantage=True,  # no advantage normalization
    learning_rate=3e-4,
    max_grad_norm=0.5,
    vf_coef=0.5,
    verbose=1,
    policy_kwargs=policy_kwargs
)


model.learn(
    total_timesteps=int(1e6),
)


model.save("PPO_FINAL")

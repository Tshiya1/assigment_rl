import os
import gym as old_gym
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import warnings
from CustomFeatureExtractors.impala import ImpalaCNN
from Wrappers.CrafterRewardWrapper import CrafterRewardWrapper

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
    
# Output directory for logs and stats
outdir = "logdir/crafter_reward-ppo/0"
os.makedirs(outdir, exist_ok=True)

# Register environment (ignore if already registered)
try:
    register(id='CrafterNoReward-v1', entry_point=crafter.Env,render_mode=None)
except Exception as e:
    print("Env already registered:", e)

print("✅ Crafter environment ready!")


env = old_gym.make('CrafterReward-v1')
env = CrafterRewardWrapper(env)
# Add recorder for stats logging
env = crafter.Recorder(
    env,
    outdir,
    save_stats=True,
    save_video=False,
    save_episode=False,
)

# Add compatibility wrapper
env = GymV21CompatibilityV0(env=env)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

policy_kwargs = dict(
    features_extractor_class=ImpalaCNN,
    features_extractor_kwargs=dict(features_dim=512),  
)


model = A2C(
    "CnnPolicy",
    env,
    learning_rate=3e-4,
    n_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    policy_kwargs=policy_kwargs,
    device="cuda"
)

model.learn(total_timesteps=1_000_000)
model.save("A2C")

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_mean_reward(json_path: str, graph_name: str, output_dir: str, window: int = 100):
    """
    Plots and saves the mean reward per window of episodes from a Crafter stats JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON-lines stats file.
    graph_name : str
        Name of the output graph image file (e.g., 'mean_reward.png').
    output_dir : str
        Folder where the graph image will be saved.
    window : int, optional
        Number of episodes per averaging window (default = 100).
    """
    # === 1. Load all JSON objects (one per line) ===
    data_list = []
    with open(json_path, "r") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    if not data_list:
        raise ValueError("No data found in JSON file.")

    print(f"Loaded {len(data_list)} game sessions")

    # === 2. Extract rewards ===
    rewards = [entry.get("reward", 0) for entry in data_list]

    # === 3. Compute mean reward per window ===
    means = [
        np.mean(rewards[i:i + window])
        for i in range(0, len(rewards), window)
    ]
    episodes = np.arange(1, len(means) + 1) * window

    # === 4. Create output directory if needed ===
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, graph_name)

    # === 5. Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, means, marker="o", color="gold")
    plt.title(f"Mean Reward per {window} Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # === 6. Save ===
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved to: {save_path}")

def plot_mean_length(json_path: str, graph_name: str, output_dir: str, window: int = 100):
    
    """
    Plots and saves the mean episode length per window of episodes from a Crafter stats JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON-lines stats file.
    graph_name : str
        Name of the output graph image file (e.g., 'mean_length.png').
    output_dir : str
        Folder where the graph image will be saved.
    window : int, optional
        Number of episodes per averaging window (default = 100).
    """
    # === 1. Load all JSON objects (one per line) ===
    data_list = []
    with open(json_path, "r") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    if not data_list:
        raise ValueError("No data found in JSON file.")

    print(f"Loaded {len(data_list)} game sessions")

    # === 2. Extract episode lengths ===
    lengths = [entry.get("length", 0) for entry in data_list]

    # === 3. Compute mean length per window ===
    means = [
        np.mean(lengths[i:i + window])
        for i in range(0, len(lengths), window)
    ]
    episodes = np.arange(1, len(means) + 1) * window

    # === 4. Create output directory if needed ===
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, graph_name)

    # === 5. Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, means, marker="o", color="dodgerblue")
    plt.title(f"Mean Episode Length per {window} Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Mean Length")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # === 6. Save ===
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved to: {save_path}")

plot_mean_length("./models/PPO/PPO_BASELINE/stats.jsonl", "PPO_BASELINE_length.png", "./media")
plot_mean_length("./models/PPO/PPO_IMPROVE_1/stats.jsonl", "PPO_IMPROVE_1_length.png", "./media")
plot_mean_length("./models/PPO/PPO_IMPROVE_2/stats.jsonl", "PPO_IMPROVE_2_length.png", "./media")
plot_mean_length("./models/A2C/A2C_BASELINE/stats.jsonl", "A2C_BASELINE_length.png", "./media")
plot_mean_length("./models/A2C/A2C_IMPROVED_1/stats.jsonl", "A2C_IMPROVE_1_length.png", "./media")
plot_mean_length("./models/A2C/A2C_IMPROVED_2/stats.jsonl", "A2C_IMPROVE_2_length.png", "./media")
import gym
from collections import defaultdict
import numpy as np


class CrafterRewardWrapper(gym.Wrapper):
    """
    Advanced reward shaping for Crafter to encourage deep progression:
    - One-time achievement bonuses
    - Tech tree milestones (Stone Age → Iron Age → Diamond)
    - Survival stat improvement
    """
    def __init__(self, env):
        super().__init__(env)

        # --- Survival tracking ---
        self.health = 9
        self.food = 9
        self.drink = 9
        self.energy = 9
        self.last_inventory = {}

        # --- Achievement tracking ---
        self.achieved = set()  # First-time events

        # --- Base achievement weights (one-time) ---
        self.achievement_weights = {
            # Collecting
            "collect_wood": 0.1,
            "collect_stone": 0.2,
            "collect_coal": 0.2,
            "collect_iron": 0.5,
            "collect_diamond": 1.5,
            "collect_sapling": 0.05,

            # Eating
            "eat_plant": 0.4,
            "eat_cow": 0.3,

            # Crafting tools
            "make_wood_pickaxe": 0.2,
            "make_wood_sword": 0.2,
            "make_stone_pickaxe": 0.4,
            "make_stone_sword": 0.4,
            "make_iron_pickaxe": 0.8,
            "make_iron_sword": 0.8,

            # Building
            "place_table": 0.5,
            "place_furnace": 0.6,
            "place_stone": 0.5,
            "place_plant": 0.1,

            # Smelting
            "make_iron_ingot": 1.0,  # Critical for iron tools

            # Combat
            "defeat_zombie": 1.2,
            "defeat_skeleton": 1.2,

            # Misc
            "wake_up": 0.5,
        }

        # --- Tech Tree Milestones (one-time big rewards) ---
        self.progression_milestones = {
            "stone_age": {"make_stone_pickaxe"},
            "iron_age": {"place_furnace", "make_iron_ingot", "make_iron_pickaxe"},
            "combat_ready": {"defeat_zombie"},
            "diamond_hunter": {"collect_diamond"},
        }

        self.milestone_rewards = {
            "stone_age": 2.0,
            "iron_age": 6.0,
            "combat_ready": 4.0,
            "diamond_hunter": 12.0,
        }


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.health = 9
        self.food = 9
        self.drink = 9
        self.energy = 9
        self.achieved = set()
        self.last_inventory = {}
        return obs


    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # --- Extract data ---
        inv = info.get("inventory", {})
        achievements = info.get("achievement", [])  # List of new achievements this step
        true_reward = reward  # Original env reward

        # --- Update survival stats ---
        prev_health = self.health
        prev_food = self.food
        prev_drink = self.drink
        prev_energy = self.energy

        self.health = inv.get("health", self.health)
        self.food = inv.get("food", self.food)
        self.drink = inv.get("drink", self.drink)
        self.energy = inv.get("energy", self.energy)

        # --- Initialize shaped reward ---
        shaped_reward = reward

        # --- 1. One-Time Achievement Bonuses ---
        for ach in achievements:
            if ach not in self.achieved:
                self.achieved.add(ach)
                weight = self.achievement_weights.get(ach, 0.0)
                if weight > 0:
                    shaped_reward += weight
                   

        # --- 2. Tech Tree Milestone Completion ---
        for milestone, required_achs in self.progression_milestones.items():
            if milestone not in self.achieved and required_achs.issubset(self.achieved):
                self.achieved.add(milestone)
                bonus = self.milestone_rewards[milestone]
                shaped_reward += bonus
                # print(f"[MILESTONE] {milestone.replace('_', ' ').title()}! (+{bonus})")

        # --- 3. Survival Stat Improvement (above baseline) ---
        shaped_reward += 0.1 * max(0, self.health - prev_health)
        shaped_reward += 0.05 * max(0, self.food - prev_food)
        shaped_reward += 0.05 * max(0, self.drink - prev_drink)
        shaped_reward += 0.05 * max(0, self.energy - prev_energy)


        # --- 4. Store for next step ---
        self.last_inventory = inv.copy()

        # --- 5. Inject true reward for logging ---
        info["true_reward"] = true_reward
        info["shaped_reward"] = shaped_reward
        info["achieved_this_step"] = achievements[:]

        return obs, shaped_reward, done, info
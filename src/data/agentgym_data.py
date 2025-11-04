"""Data collection for AgentGym tasks (RL environments)"""

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from test_code import environments as test_envs
except ImportError:
    test_envs = None
    print("Warning: test-code module not found. Using synthetic data generation.")


@dataclass
class RLEpisode:
    """Represents a complete RL episode"""
    environment: str
    observations: List[str]
    actions: List[str]
    rewards: List[float]
    dones: List[bool]
    total_reward: float
    episode_length: int
    metadata: Dict[str, Any]


@dataclass
class RLTransition:
    """Represents a single transition in an RL episode"""
    observation: str
    action: str
    reward: float
    next_observation: str
    done: bool
    environment: str


class AgentGymDataCollector:
    """Collects and processes data for AgentGym RL tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_episodes = config.get("num_episodes", 5000)
        self.max_episode_length = config.get("max_episode_length", 50)
        self.environments = config.get("environments", [
            "agentgym:webshop",
            "agentgym:alfworld",
            "agentgym:babyai",
            "agentgym:sciworld",
            "agentgym:textcraft"
        ])

    async def collect_webshop_episodes(self, num_episodes: int) -> List[RLEpisode]:
        """Collect WebShop environment episodes"""
        episodes = []

        products = [
            "wireless headphones", "laptop bag", "water bottle",
            "phone case", "desk lamp", "notebook", "pen set"
        ]

        for i in range(num_episodes):
            target_product = random.choice(products)
            observations = [f"[WebShop] Find and purchase: {target_product}"]
            actions = []
            rewards = []
            dones = []

            # Simulate search and purchase
            actions.append(f"search[{target_product}]")
            observations.append(f"[Search Results] Showing results for '{target_product}'")
            rewards.append(0.1)
            dones.append(False)

            # Click on item
            item_idx = random.randint(1, 5)
            actions.append(f"click[{item_idx}]")
            observations.append(f"[Product Page] {target_product} - $29.99")
            rewards.append(0.3)
            dones.append(False)

            # Purchase
            actions.append("click[Buy Now]")
            observations.append("[Success] Purchase completed!")
            rewards.append(1.0)
            dones.append(True)

            episodes.append(RLEpisode(
                environment="agentgym:webshop",
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                total_reward=sum(rewards),
                episode_length=len(actions),
                metadata={"target": target_product}
            ))

        return episodes

    async def collect_alfworld_episodes(self, num_episodes: int) -> List[RLEpisode]:
        """Collect AlfWorld environment episodes"""
        episodes = []

        tasks = [
            ("put a clean mug in the coffee machine", ["goto[coffee machine]", "take[mug 1]", "goto[sink]", "clean[mug 1]", "goto[coffee machine]", "put[mug 1]"]),
            ("put a hot potato in the fridge", ["goto[stove]", "take[potato 1]", "heat[potato 1]", "goto[fridge]", "put[potato 1]"]),
            ("examine a book with the lamp", ["goto[shelf]", "take[book 1]", "goto[desk]", "use[desk lamp 1]", "examine[book 1]"]),
        ]

        for i in range(num_episodes):
            task_desc, task_actions = random.choice(tasks)
            observations = [f"[AlfWorld] Task: {task_desc}"]
            actions = []
            rewards = []
            dones = []

            for action in task_actions:
                actions.append(action)
                observations.append(f"You {action}")
                rewards.append(0.2)
                dones.append(False)

            # Final reward
            rewards[-1] = 1.0
            dones[-1] = True
            observations.append(f"Task completed successfully!")

            episodes.append(RLEpisode(
                environment="agentgym:alfworld",
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                total_reward=sum(rewards),
                episode_length=len(actions),
                metadata={"task": task_desc}
            ))

        return episodes

    async def collect_babyai_episodes(self, num_episodes: int) -> List[RLEpisode]:
        """Collect BabyAI environment episodes"""
        episodes = []

        goals = [
            ("go to the red ball", ["turn right", "move forward", "move forward", "turn left", "move forward"]),
            ("pick up the blue key", ["move forward", "turn right", "move forward", "pick up"]),
            ("open the door", ["goto[door]", "toggle[door]", "go through[door]"]),
        ]

        for i in range(num_episodes):
            goal_desc, goal_actions = random.choice(goals)
            observations = [f"[BabyAI] Goal: {goal_desc}\nYou see a 7x7 grid."]
            actions = []
            rewards = []
            dones = []

            for action in goal_actions:
                actions.append(action)
                observations.append(f"Action executed: {action}")
                rewards.append(0.0)
                dones.append(False)

            # Final reward
            rewards[-1] = 1.0
            dones[-1] = True
            observations.append(f"Goal achieved!")

            episodes.append(RLEpisode(
                environment="agentgym:babyai",
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                total_reward=sum(rewards),
                episode_length=len(actions),
                metadata={"goal": goal_desc}
            ))

        return episodes

    async def collect_sciworld_episodes(self, num_episodes: int) -> List[RLEpisode]:
        """Collect SciWorld environment episodes"""
        episodes = []

        experiments = [
            ("measure the temperature of boiling water", ["goto[stove]", "take[beaker]", "pour[water, beaker]", "activate[stove]", "use[thermometer, beaker]"]),
            ("grow a plant", ["goto[greenhouse]", "take[seed]", "goto[pot]", "put[seed, pot]", "use[watering can, pot]"]),
        ]

        for i in range(num_episodes):
            exp_desc, exp_actions = random.choice(experiments)
            observations = [f"[SciWorld] Task: {exp_desc}"]
            actions = []
            rewards = []
            dones = []

            for action in exp_actions:
                actions.append(action)
                observations.append(f"Action: {action}")
                rewards.append(0.15)
                dones.append(False)

            # Final reward
            rewards[-1] = 1.0
            dones[-1] = True
            observations.append(f"Experiment completed!")

            episodes.append(RLEpisode(
                environment="agentgym:sciworld",
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                total_reward=sum(rewards),
                episode_length=len(actions),
                metadata={"experiment": exp_desc}
            ))

        return episodes

    async def collect_textcraft_episodes(self, num_episodes: int) -> List[RLEpisode]:
        """Collect TextCraft environment episodes - expanded with diverse recipes"""
        episodes = []

        # Expanded recipe library with diverse crafting patterns
        recipes = [
            # Basic tools
            ("craft a pickaxe", ["get[wood]", "craft[planks]", "craft[sticks]", "get[cobblestone]", "craft[pickaxe]"]),
            ("craft a sword", ["get[wood]", "craft[planks]", "craft[sticks]", "get[iron ore]", "craft[sword]"]),
            ("craft an axe", ["get[wood]", "craft[planks]", "craft[sticks]", "get[cobblestone]", "craft[axe]"]),
            ("craft a shovel", ["get[wood]", "craft[planks]", "craft[sticks]", "get[iron ore]", "craft[shovel]"]),
            ("craft a hoe", ["get[wood]", "craft[planks]", "craft[sticks]", "get[iron ore]", "craft[hoe]"]),
            
            # Weapons
            ("craft a bow", ["get[wood]", "craft[planks]", "craft[sticks]", "get[string]", "craft[bow]"]),
            ("craft arrows", ["get[sticks]", "get[flint]", "get[feather]", "craft[arrows]", "craft[arrows]", "craft[arrows]"]),
            ("craft a shield", ["get[wood]", "craft[planks]", "craft[planks]", "craft[planks]", "get[iron ingot]", "craft[shield]"]),
            ("craft a dagger", ["get[iron ingot]", "craft[stick]", "craft[dagger]"]),
            ("craft a spear", ["get[iron ingot]", "get[stick]", "craft[spear]"]),
            
            # Building materials
            ("craft wooden planks", ["get[wood]", "craft[planks]", "craft[planks]", "craft[planks]", "craft[planks]"]),
            ("craft sticks", ["get[planks]", "craft[sticks]", "craft[sticks]"]),
            ("craft stone bricks", ["get[cobblestone]", "craft[stone bricks]", "craft[stone bricks]"]),
            ("craft glass", ["get[sand]", "smelt[glass]"]),
            ("craft torches", ["get[stick]", "get[coal]", "craft[torch]", "craft[torch]", "craft[torch]", "craft[torch]"]),
            
            # Furnace and smelting
            ("craft a furnace", ["get[cobblestone]", "get[cobblestone]", "get[cobblestone]", "get[cobblestone]", "get[cobblestone]", "get[cobblestone]", "get[cobblestone]", "get[cobblestone]", "craft[furnace]"]),
            ("smelt iron ingot", ["get[iron ore]", "get[coal]", "smelt[iron ingot]"]),
            ("smelt gold ingot", ["get[gold ore]", "get[coal]", "smelt[gold ingot]"]),
            ("smelt glass", ["get[sand]", "smelt[glass]"]),
            ("cook food", ["get[raw meat]", "cook[steak]"]),
            
            # Armor
            ("craft leather helmet", ["get[leather]", "get[leather]", "get[leather]", "craft[leather helmet]"]),
            ("craft iron chestplate", ["get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "craft[iron chestplate]"]),
            ("craft iron boots", ["get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "get[iron ingot]", "craft[iron boots]"]),
            
            # Food and consumables
            ("craft bread", ["get[wheat]", "get[wheat]", "get[wheat]", "craft[bread]"]),
            ("craft cake", ["get[milk]", "get[egg]", "get[wheat]", "get[sugar]", "craft[cake]"]),
            ("craft cooked chicken", ["get[raw chicken]", "cook[cooked chicken]"]),
            
            # Redstone and advanced
            ("craft redstone torch", ["get[stick]", "get[redstone]", "craft[redstone torch]"]),
            ("craft lever", ["get[stick]", "get[cobblestone]", "craft[lever]"]),
            ("craft button", ["get[stone]", "craft[button]"]),
            
            # Decorative items
            ("craft painting", ["get[wood]", "craft[planks]", "get[wool]", "craft[painting]"]),
            ("craft bookshelf", ["get[planks]", "get[planks]", "get[planks]", "get[book]", "get[book]", "get[book]", "craft[bookshelf]"]),
            
            # Complex multi-step recipes
            ("craft iron pickaxe", ["get[iron ore]", "get[coal]", "smelt[iron ingot]", "get[iron ingot]", "get[iron ingot]", "get[stick]", "craft[iron pickaxe]"]),
            ("craft diamond sword", ["get[diamond]", "get[diamond]", "get[stick]", "craft[diamond sword]"]),
            ("craft enchanted book", ["get[book]", "get[lapis lazuli]", "get[experience]", "enchant[enchanted book]"]),
            
            # Utility items
            ("craft a chest", ["get[planks]", "get[planks]", "get[planks]", "get[planks]", "get[planks]", "get[planks]", "get[planks]", "get[planks]", "craft[chest]"]),
            ("craft crafting table", ["get[planks]", "get[planks]", "get[planks]", "get[planks]", "craft[crafting table]"]),
            ("craft ladder", ["get[stick]", "get[stick]", "get[stick]", "get[stick]", "get[stick]", "get[stick]", "get[stick]", "craft[ladder]"]),
        ]

        for i in range(num_episodes):
            recipe_desc, recipe_actions = random.choice(recipes)
            
            # Add some variation: sometimes inventory starts with some items
            starting_inventory = []
            if random.random() < 0.3:  # 30% chance to start with some items
                available_items = ["wood", "planks", "stick", "cobblestone", "iron ore", "coal"]
                starting_items = random.sample(available_items, random.randint(1, 2))
                starting_inventory = starting_items
            
            # Add randomness: sometimes take extra steps or explore
            observations = [f"[TextCraft] Goal: {recipe_desc}\nInventory: {starting_inventory if starting_inventory else '[]'}"]
            actions = []
            rewards = []
            dones = []
            
            inventory_items = set(starting_inventory)
            
            for action in recipe_actions:
                # Sometimes add exploration actions (10% chance)
                if random.random() < 0.1 and len(actions) > 0:
                    explore_actions = ["explore[north]", "explore[south]", "explore[east]", "explore[west]"]
                    explore_action = random.choice(explore_actions)
                    actions.append(explore_action)
                    observations.append(f"Action: {explore_action}\nNo items found.")
                    rewards.append(0.0)  # No reward for exploration
                    dones.append(False)
                
                # Add the actual recipe action
                actions.append(action)
                
                # Parse action to update inventory
                if action.startswith("get["):
                    item = action[4:-1]
                    inventory_items.add(item)
                    observations.append(f"Action: {action}\nInventory: {list(inventory_items)}")
                elif action.startswith("craft[") or action.startswith("smelt[") or action.startswith("cook["):
                    item = action.split("[")[1][:-1]
                    inventory_items.add(item)
                    observations.append(f"Action: {action}\nInventory: {list(inventory_items)}")
                else:
                    observations.append(f"Action: {action}\nInventory: {list(inventory_items)}")
                
                rewards.append(0.15)
                dones.append(False)

            # Final reward
            rewards[-1] = 1.0
            dones[-1] = True
            observations.append(f"Successfully crafted {recipe_desc}!")

            episodes.append(RLEpisode(
                environment="agentgym:textcraft",
                observations=observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
                total_reward=sum(rewards),
                episode_length=len(actions),
                metadata={"recipe": recipe_desc, "starting_inventory": starting_inventory}
            ))

        return episodes

    async def collect_all_episodes(self) -> Dict[str, List[RLEpisode]]:
        """Collect episodes for all AgentGym environments"""
        all_episodes = {}
        episodes_per_env = self.num_episodes // len(self.environments)

        for env in self.environments:
            print(f"Collecting {episodes_per_env} episodes for {env}...")

            if env == "agentgym:webshop":
                episodes = await self.collect_webshop_episodes(episodes_per_env)
            elif env == "agentgym:alfworld":
                episodes = await self.collect_alfworld_episodes(episodes_per_env)
            elif env == "agentgym:babyai":
                episodes = await self.collect_babyai_episodes(episodes_per_env)
            elif env == "agentgym:sciworld":
                episodes = await self.collect_sciworld_episodes(episodes_per_env)
            elif env == "agentgym:textcraft":
                episodes = await self.collect_textcraft_episodes(episodes_per_env)
            else:
                print(f"Unknown environment: {env}")
                continue

            all_episodes[env] = episodes
            print(f"Collected {len(episodes)} episodes for {env}")

        return all_episodes

    def episodes_to_transitions(self, episodes: List[RLEpisode]) -> List[RLTransition]:
        """Convert episodes to transitions for training"""
        transitions = []

        for episode in episodes:
            for i in range(len(episode.actions)):
                transition = RLTransition(
                    observation=episode.observations[i],
                    action=episode.actions[i],
                    reward=episode.rewards[i],
                    next_observation=episode.observations[i + 1] if i + 1 < len(episode.observations) else "",
                    done=episode.dones[i],
                    environment=episode.environment
                )
                transitions.append(transition)

        return transitions

    def save_episodes(self, episodes: Dict[str, List[RLEpisode]], output_dir: Path):
        """Save collected episodes to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for env, env_episodes in episodes.items():
            env_name = env.replace(":", "_")
            output_file = output_dir / f"{env_name}_episodes.jsonl"

            with open(output_file, 'w') as f:
                for episode in env_episodes:
                    json_line = asdict(episode)
                    f.write(json.dumps(json_line) + '\n')

            print(f"Saved {len(env_episodes)} episodes to {output_file}")

    def load_episodes(self, input_dir: Path) -> Dict[str, List[RLEpisode]]:
        """Load episodes from disk"""
        all_episodes = {}

        for env in self.environments:
            env_name = env.replace(":", "_")
            input_file = input_dir / f"{env_name}_episodes.jsonl"

            if not input_file.exists():
                print(f"Warning: {input_file} not found")
                continue

            episodes = []
            with open(input_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    episodes.append(RLEpisode(**data))

            all_episodes[env] = episodes
            print(f"Loaded {len(episodes)} episodes from {input_file}")

        return all_episodes


async def main():
    """Example usage"""
    config = {
        "num_episodes": 100,
        "max_episode_length": 50,
        "environments": [
            "agentgym:webshop",
            "agentgym:alfworld",
            "agentgym:babyai",
            "agentgym:sciworld",
            "agentgym:textcraft"
        ]
    }

    collector = AgentGymDataCollector(config)
    episodes = await collector.collect_all_episodes()

    output_dir = Path("data_cache/agentgym")
    collector.save_episodes(episodes, output_dir)


if __name__ == "__main__":
    asyncio.run(main())

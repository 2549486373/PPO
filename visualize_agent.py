"""
Visualize a trained PPO agent in the environment.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def load_agent(model_path, env_name="CartPole-v1"):
    """Load a trained agent from a model file."""
    
    # Environment parameters
    if env_name == "CartPole-v1":
        state_dim, action_dim = 4, 2
    elif env_name == "LunarLander-v2":
        state_dim, action_dim = 8, 4
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    
    # Create agent with same architecture as training
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device=PPOConfig.DEVICE
    )
    
    # Load the trained model
    try:
        agent.load(model_path)
        print(f"✅ Agent loaded successfully from {model_path}")
        return agent
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def visualize_agent(agent, env_name="CartPole-v1", num_episodes=5, render=True, save_video=False, video_path=None):
    """Visualize the agent in the environment."""
    
    if agent is None:
        print("❌ No agent provided for visualization")
        return
    
    # Create trainer for environment interaction
    trainer = PPOTrainer(
        agent=agent,
        env_name=env_name,
        config={
            'max_episode_length': PPOConfig.MAX_EPISODE_LENGTH
        }
    )
    
    # Visualize the agent
    trainer.visualize_agent(
        num_episodes=num_episodes,
        render=render,
        save_video=save_video,
        video_path=video_path or f"videos/{env_name}_visualization.mp4"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained PPO agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--save-video", action="store_true", help="Save visualization as video")
    parser.add_argument("--video-path", type=str, help="Path to save video file")
    
    args = parser.parse_args()
    
    print(f"🎬 Loading trained agent from: {args.model}")
    
    # Load the agent
    agent = load_agent(args.model, args.env)
    
    if agent:
        # Visualize the agent
        visualize_agent(
            agent=agent,
            env_name=args.env,
            num_episodes=args.episodes,
            render=not args.no_render,
            save_video=args.save_video,
            video_path=args.video_path
        )
    else:
        print("❌ Failed to load agent")


if __name__ == "__main__":
    main() 
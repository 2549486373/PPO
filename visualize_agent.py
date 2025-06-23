"""
Visualize trained PPO agents in their environments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import gymnasium as gym
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer


def visualize_trained_agent(model_path: str, env_name: str = "CartPole-v1", 
                           num_episodes: int = 5, render: bool = True, 
                           save_video: bool = False, video_path: str = None):
    """
    Visualize a trained PPO agent.
    
    Args:
        model_path: Path to the saved model (.pth file)
        env_name: Name of the environment
        num_episodes: Number of episodes to visualize
        render: Whether to render the environment
        save_video: Whether to save the visualization as video
        video_path: Path to save the video file
    """
    print(f"🎬 Loading trained agent from: {model_path}")
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        learning_rate=3e-4,
        device=device
    )
    
    # Load trained model
    try:
        agent.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create trainer for visualization
    trainer = PPOTrainer(
        agent=agent,
        env=env,
        config={
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'batch_size': 64,
            'num_epochs': 10,
            'buffer_size': 2048
        }
    )
    
    # Set video path if not provided
    if save_video and video_path is None:
        video_path = f"videos/{env_name.lower()}_visualization.mp4"
    
    # Visualize the agent
    results = trainer.visualize_agent(
        num_episodes=num_episodes,
        render=render,
        save_video=save_video,
        video_path=video_path
    )
    
    env.close()
    return results


def main():
    """Main function for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trained PPO agent")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to the trained model (.pth file)")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                       help="Environment name")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to visualize")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering (for headless environments)")
    parser.add_argument("--save-video", action="store_true",
                       help="Save visualization as video")
    parser.add_argument("--video-path", type=str, default=None,
                       help="Path to save video file")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Visualize the agent
    results = visualize_trained_agent(
        model_path=args.model,
        env_name=args.env,
        num_episodes=args.episodes,
        render=not args.no_render,
        save_video=args.save_video,
        video_path=args.video_path
    )
    
    if results:
        print(f"\n🎉 Visualization completed!")
        print(f"Average reward: {results['avg_reward']:.2f}")
        print(f"Average episode length: {results['avg_length']:.1f}")


if __name__ == "__main__":
    main() 
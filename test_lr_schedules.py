"""
Test script to demonstrate different learning rate scheduling options for PPO.
This script shows how different LR schedules affect training performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
from ppo.agent import PPOAgent
from ppo.trainer import PPOTrainer
from configs.default_config import PPOConfig


def test_lr_schedule(schedule_type, schedule_kwargs, episodes=200, env_name="CartPole-v1"):
    """Test a specific learning rate schedule."""
    
    print(f"\n🧪 Testing {schedule_type} learning rate schedule...")
    print(f"   Schedule kwargs: {schedule_kwargs}")
    
    # Environment parameters
    state_dim = 4 if env_name == "CartPole-v1" else 8
    action_dim = 2 if env_name == "CartPole-v1" else 4
    
    # Create agent with specific LR schedule
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=PPOConfig.HIDDEN_DIM,  # Use config value
        num_layers=PPOConfig.NUM_LAYERS,
        learning_rate=PPOConfig.LEARNING_RATE,
        device="cpu",
        lr_schedule=schedule_type,
        lr_schedule_kwargs=schedule_kwargs
    )
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        env_name=env_name,
        config={
            'gamma': PPOConfig.GAMMA,
            'gae_lambda': PPOConfig.GAE_LAMBDA,
            'clip_epsilon': PPOConfig.CLIP_EPSILON,
            'value_coef': PPOConfig.VALUE_COEF,
            'entropy_coef': PPOConfig.ENTROPY_COEF,
            'max_grad_norm': PPOConfig.MAX_GRAD_NORM,
            'batch_size': 32,
            'num_epochs': 5,
            'buffer_size': 512,
            'target_kl': PPOConfig.TARGET_KL,
            'max_episode_length': 500,
            'log_interval': 10,
            'save_interval': 50,
            'eval_interval': 15,
            'print_interval': 25,
            'early_stopping_patience': 0,  # Disabled early stopping for comparison
            'early_stopping_threshold': 0.0  # Not used when early stopping is disabled
        }
    )
    
    # Setup logging
    log_dir = f"runs/test_lr_{schedule_type}"
    trainer.setup_logging(log_dir)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Train the agent
    save_path = f"models/test_{schedule_type}.pth"
    trainer.train(episodes=episodes, save_path=save_path)
    
    # Get training history
    recent_episodes = trainer.episode_buffer.get_recent_episodes(episodes)
    rewards = [ep['reward'] for ep in recent_episodes]
    
    # Get learning rate history
    lr_info = agent.get_lr_schedule_info()
    
    return {
        'schedule_type': schedule_type,
        'rewards': rewards,
        'final_lr': agent.get_current_lr(),
        'initial_lr': lr_info['initial_lr'],
        'log_dir': log_dir
    }


def plot_lr_comparison(results):
    """Plot comparison of different learning rate schedules."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reward curves
    ax1 = axes[0, 0]
    for result in results:
        rewards = result['rewards']
        episodes = list(range(len(rewards)))
        ax1.plot(episodes, rewards, label=f"{result['schedule_type']}", alpha=0.8)
    
    ax1.set_title('Training Rewards by LR Schedule')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average rewards
    ax2 = axes[0, 1]
    window = 20
    for result in results:
        rewards = result['rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = list(range(window-1, len(rewards)))
            ax2.plot(episodes, moving_avg, label=f"{result['schedule_type']}", alpha=0.8)
    
    ax2.set_title(f'Moving Average Rewards (window={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Moving Avg Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final performance comparison
    ax3 = axes[1, 0]
    schedule_types = [r['schedule_type'] for r in results]
    final_rewards = [np.mean(r['rewards'][-50:]) if len(r['rewards']) >= 50 else np.mean(r['rewards']) for r in results]
    
    bars = ax3.bar(schedule_types, final_rewards, alpha=0.8)
    ax3.set_title('Final Performance (Last 50 Episodes Avg)')
    ax3.set_ylabel('Average Reward')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_rewards):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    # Plot 4: Learning rate decay visualization
    ax4 = axes[1, 1]
    for result in results:
        # Simulate LR decay for visualization
        initial_lr = result['initial_lr']
        final_lr = result['final_lr']
        schedule_type = result['schedule_type']
        
        episodes = list(range(200))  # Assume 200 episodes
        
        if schedule_type == "linear":
            lrs = [initial_lr * (1 - i/200) for i in episodes]
        elif schedule_type == "cosine":
            lrs = [initial_lr * 0.5 * (1 + np.cos(np.pi * i / 200)) for i in episodes]
        elif schedule_type == "exponential":
            lrs = [initial_lr * (0.99 ** i) for i in episodes]
        elif schedule_type == "step":
            lrs = [initial_lr * (0.1 ** (i // 50)) for i in episodes]
        else:  # none
            lrs = [initial_lr] * len(episodes)
        
        ax4.plot(episodes, lrs, label=f"{schedule_type}", alpha=0.8)
    
    ax4.set_title('Learning Rate Decay Patterns')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/lr_schedule_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Main function to test different learning rate schedules."""
    
    print("🧪 Testing Different Learning Rate Schedules")
    print("=" * 50)
    
    # Define different LR schedules to test
    lr_schedules = [
        {
            'type': 'none',
            'kwargs': {},
            'description': 'Constant learning rate'
        },
        {
            'type': 'linear',
            'kwargs': {'total_iters': 200, 'end_factor': 0.0},
            'description': 'Linear decay to 0'
        },
        {
            'type': 'cosine',
            'kwargs': {'total_iters': 200, 'eta_min': 1e-6},
            'description': 'Cosine annealing'
        },
        {
            'type': 'exponential',
            'kwargs': {'gamma': 0.995},
            'description': 'Exponential decay'
        },
        {
            'type': 'step',
            'kwargs': {'step_size': 50, 'gamma': 0.5},
            'description': 'Step decay every 50 episodes'
        }
    ]
    
    results = []
    
    # Test each schedule
    for schedule in lr_schedules:
        try:
            result = test_lr_schedule(
                schedule['type'], 
                schedule['kwargs'], 
                episodes=200,
                env_name="CartPole-v1"
            )
            results.append(result)
            print(f"✅ {schedule['type']}: Final LR = {result['final_lr']:.2e}")
        except Exception as e:
            print(f"❌ {schedule['type']}: Failed - {e}")
    
    # Plot comparison
    if results:
        print(f"\n📊 Plotting comparison of {len(results)} schedules...")
        plot_lr_comparison(results)
        
        # Print summary
        print(f"\n📈 Performance Summary:")
        print("-" * 40)
        for result in results:
            final_avg = np.mean(result['rewards'][-50:]) if len(result['rewards']) >= 50 else np.mean(result['rewards'])
            print(f"{result['schedule_type']:12}: Final Avg = {final_avg:.1f}, Final LR = {result['final_lr']:.2e}")
    
    print(f"\n✅ Test completed! Check plots/lr_schedule_comparison.png for visualization.")


if __name__ == "__main__":
    main() 
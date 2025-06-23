import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

from .agent import PPOAgent
from .memory import PPOMemory, EpisodeBuffer
from .utils import normalize

class PPOTrainer:
    """
    PPO Trainer for managing training loops and environment interaction.
    """
    
    def __init__(self, agent: PPOAgent, env_name: Optional[str] = None, env: Optional[gym.Env] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.agent = agent
        self.env = env if env else gym.make(env_name)
        self.config = config or {}

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_epochs = self.config.get('num_epochs', 10)
        self.buffer_size = self.config.get('buffer_size', 2048)
        self.target_kl = self.config.get('target_kl', 0.01)
        self.max_episode_length = self.config.get('max_episode_length', 1000)

        self.memory = PPOMemory(buffer_size=self.buffer_size)
        self.episode_buffer = EpisodeBuffer(max_episodes=100)
        self.writer = None

        self.log_interval = self.config.get('log_interval', 10)
        self.save_interval = self.config.get('save_interval', 100)
        self.eval_interval = self.config.get('eval_interval', 50)
        self.print_interval = self.config.get('print_interval', 100)  # Print rewards every N epochs
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.epoch_count = 0  # Track total epochs for plotting
        
        # Early stopping parameters
        self.early_stopping_patience = self.config.get('early_stopping_patience', 50)
        self.early_stopping_threshold = self.config.get('early_stopping_threshold', 0.0)
        self.early_stopping_counter = 0
        self.early_stopping_best_reward = float('-inf')

    def setup_logging(self, log_dir: str = "runs/ppo"):
        self.writer = SummaryWriter(log_dir)

    def collect_episode(self, deterministic: bool = False) -> Dict[str, Any]:
        state, _ = self.env.reset()
        done, truncated = False, False
        episode_reward, episode_length = 0, 0

        while not (done or truncated) and episode_length < self.max_episode_length:
            action, log_prob, value = self.agent.get_action(state, deterministic)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            self.memory.store(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1

        ep_data = {'reward': episode_reward, 'length': episode_length}
        self.episode_buffer.store_episode(ep_data)
        return ep_data

    def collect_batch(self) -> int:
        self.memory.reset()
        num_episodes = 0

        while len(self.memory) < self.buffer_size:
            ep_data = self.collect_episode()
            num_episodes += 1
            self.episode_count += 1
            
            if self.writer and self.episode_count % self.log_interval == 0:
                self.writer.add_scalar('Episode/Reward', ep_data['reward'], self.episode_count)
                self.writer.add_scalar('Episode/Length', ep_data['length'], self.episode_count)

        # Get last state for bootstrapping value
        state, _ = self.env.reset()
        _, _, last_value = self.agent.get_action(state, deterministic=True)
        self.memory.compute_advantages(self.gamma, self.gae_lambda, last_value)
        return num_episodes

    def update_policy(self) -> Dict[str, float]:
        total_loss_info = {}
        
        for epoch in range(self.num_epochs):
            batch = self.memory.get_batch(self.batch_size)
            loss_info = self.agent.update(batch, self.clip_epsilon,
                                           self.value_coef, self.entropy_coef, self.max_grad_norm)
            for k, v in loss_info.items():
                total_loss_info.setdefault(k, []).append(v)
            self.epoch_count += 1  # Track total epochs

            if loss_info.get('kl_div', 0) > self.target_kl:
                break
        
        return {k: np.mean(v) for k, v in total_loss_info.items()}

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        rewards, lengths = [], []
        for _ in range(num_episodes):
            ep = self.collect_episode(deterministic=True)
            rewards.append(ep['reward'])
            lengths.append(ep['length'])

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths)
        }

    def train(self, episodes: int = 1000, save_path: Optional[str] = None):
        total_epochs = episodes * self.num_epochs
        print(f"Starting PPO training for {episodes} episodes ({total_epochs} total epochs)...")
        
        # Print learning rate schedule information
        lr_info = self.agent.get_lr_schedule_info()
        print(f"Learning rate schedule: {lr_info['schedule_type']}")
        print(f"Initial learning rate: {lr_info['initial_lr']:.2e}")
        print(f"Epochs per episode: {self.num_epochs}")
        print(f"Expected total epochs: {total_epochs}")
        
        if self.writer is None:
            self.setup_logging()

        for episode in tqdm(range(episodes), desc="Training"):
            self.collect_batch()
            loss_info = self.update_policy()

            if self.writer:
                for k, v in loss_info.items():
                    self.writer.add_scalar(f'Loss/{k}', v, self.episode_count)
                stats = self.episode_buffer.get_episode_stats()
                for k, v in stats.items():
                    self.writer.add_scalar(f'Stats/{k}', v, self.episode_count)
                
                # Log learning rate and epoch information
                self.writer.add_scalar('Training/LearningRate', loss_info['learning_rate'], self.episode_count)
                self.writer.add_scalar('Training/TotalEpochs', self.epoch_count, self.episode_count)
                self.writer.add_scalar('Training/EpochsPerEpisode', self.num_epochs, self.episode_count)

            # Print rewards every print_interval episodes
            if episode % self.print_interval == 0:
                stats = self.episode_buffer.get_episode_stats()
                recent_rewards = [ep['reward'] for ep in self.episode_buffer.get_recent_episodes(50)]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                current_lr = loss_info['learning_rate']
                print(f"Episode {episode} (Epochs {episode * self.num_epochs}-{(episode + 1) * self.num_epochs - 1}): "
                      f"Avg Reward (last 50): {avg_reward:.2f}, "
                      f"Best Reward: {self.best_reward:.2f}, LR: {current_lr:.2e}")

            # Save model every save_interval episodes
            if save_path and episode % self.save_interval == 0:
                checkpoint_path = save_path.replace('.pth', f'_episode_{episode}.pth')
                self.agent.save(checkpoint_path)
                print(f"Model saved at episode {episode}: {checkpoint_path}")

            if episode % self.eval_interval == 0:
                eval_info = self.evaluate()
                if self.writer:
                    for k, v in eval_info.items():
                        self.writer.add_scalar(f'Eval/{k}', v, self.episode_count)

                if save_path and eval_info['mean_reward'] > self.best_reward:
                    self.best_reward = eval_info['mean_reward']
                    self.agent.save(save_path)
                    print(f"New best model saved! Reward: {self.best_reward:.2f}")

            # Early stopping check
            if self.early_stopping_patience > 0:
                current_reward = self.best_reward
                if current_reward > self.early_stopping_best_reward + self.early_stopping_threshold:
                    self.early_stopping_best_reward = current_reward
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered! No improvement for {self.early_stopping_patience} evaluations.")
                    print(f"Best reward achieved: {self.early_stopping_best_reward:.2f}")
                    break

        print("Training completed!")
        if self.writer:
            self.writer.close()

    def plot_training_curves(self, save_path: Optional[str] = None):
        stats = self.episode_buffer.get_episode_stats()
        if not stats:
            print("No data to plot.")
            return

        recent = self.episode_buffer.get_recent_episodes(100)
        rewards = [ep['reward'] for ep in recent]
        
        # Calculate epochs for each episode (each episode has num_epochs policy updates)
        epochs_per_episode = self.num_epochs
        total_epochs = len(rewards) * epochs_per_episode
        epochs = list(range(0, total_epochs, epochs_per_episode))  # Epoch numbers: 0, 10, 20, 30, ...

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, rewards)
        plt.title('Training Reward vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        window = min(20, len(rewards))
        if window > 0:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_avg_epochs = epochs[window-1:]  # Adjust epochs for moving average
            plt.plot(moving_avg_epochs, moving_avg)
            plt.title(f'Moving Avg (window={window}) vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Moving Average Reward')
            plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.hist(rewards, bins=20, alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        keys = ['mean_reward', 'std_reward', 'min_reward', 'max_reward']
        values = [stats.get(k, 0) for k in keys]
        plt.bar(keys, values)
        plt.title('Statistics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.tight_layout()
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

    def visualize_agent(self, num_episodes: int = 5, render: bool = True, save_video: bool = False, video_path: str = "videos/agent_visualization.mp4"):
        """
        Visualize the trained agent in the environment.
        
        Args:
            num_episodes: Number of episodes to visualize
            render: Whether to render the environment
            save_video: Whether to save the visualization as video
            video_path: Path to save the video file
        """
        if save_video:
            # Create video directory if it doesn't exist
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # Import OpenCV for video recording
            try:
                import cv2
            except ImportError:
                print("❌ OpenCV not installed. Install with: pip install opencv-python")
                save_video = False
            
            # Create a new environment with video recording
            try:
                env = gym.make(self.env.spec.id, render_mode="rgb_array")
            except Exception as e:
                if "pygame" in str(e).lower():
                    print("❌ Pygame not installed. Install with: pip install pygame")
                    print("   Or install all dependencies: pip install gymnasium[classic-control]")
                else:
                    print(f"❌ Error creating environment: {e}")
                return None
            
            # Setup video writer
            if save_video:
                try:
                    # Get first frame to determine video dimensions
                    test_state, _ = env.reset()
                    test_frame = env.render()
                    height, width, _ = test_frame.shape
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                    print(f"📹 Recording video to: {video_path}")
                except Exception as e:
                    print(f"❌ Error setting up video recording: {e}")
                    save_video = False
                
        elif render:
            # Create a new environment with rendering
            try:
                env = gym.make(self.env.spec.id, render_mode="human")
            except Exception as e:
                if "pygame" in str(e).lower():
                    print("❌ Pygame not installed. Install with: pip install pygame")
                    print("   Or install all dependencies: pip install gymnasium[classic-control]")
                else:
                    print(f"❌ Error creating environment: {e}")
                return None
        else:
            try:
                env = gym.make(self.env.spec.id)
            except Exception as e:
                print(f"❌ Error creating environment: {e}")
                return None
        
        print(f"\n🎬 Visualizing trained agent for {num_episodes} episodes...")
        
        total_reward = 0
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done, truncated = False, False
            
            print(f"Episode {episode + 1}: ", end="")
            
            while not (done or truncated) and episode_length < self.max_episode_length:
                # Get action from trained agent (deterministic for visualization)
                action, _, _ = self.agent.get_action(state, deterministic=True)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Record frame if saving video
                if save_video:
                    try:
                        frame = env.render()
                        # Convert from RGB to BGR (OpenCV format)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                    except Exception as e:
                        print(f"❌ Error recording frame: {e}")
                        save_video = False
                        break
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Add small delay for better visualization
                if render and not save_video:
                    import time
                    time.sleep(0.01)
            
            total_reward += episode_reward
            episode_lengths.append(episode_length)
            print(f"Reward: {episode_reward:.1f}, Length: {episode_length}")
        
        # Close video writer if recording
        if save_video:
            try:
                video_writer.release()
                print(f"✅ Video saved successfully to: {video_path}")
            except Exception as e:
                print(f"❌ Error closing video writer: {e}")
        
        env.close()
        
        avg_reward = total_reward / num_episodes
        avg_length = sum(episode_lengths) / len(episode_lengths)
        
        print(f"\n📊 Visualization Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'episode_rewards': [total_reward / num_episodes] * num_episodes,
            'episode_lengths': episode_lengths
        }

    def get_epoch_info(self):
        """Get information about epoch tracking."""
        return {
            'total_epochs': self.epoch_count,
            'total_episodes': self.episode_count,
            'epochs_per_episode': self.num_epochs,
            'current_episode_epochs': self.epoch_count % self.num_epochs
        }

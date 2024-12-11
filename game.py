import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Define constants
GRID_SIZE = 20
WINDOW_SIZE = 400
TILE_SIZE = WINDOW_SIZE // GRID_SIZE

# Rewards
REWARD_PELLET = 20
REWARD_SUPER_PELLET = 100
REWARD_GHOST = -200
REWARD_STEP = -1

# Initialize pygame
pygame.init()

class PacManEnv:
    def __init__(self):
        self.grid = self.create_grid()
        self.pacman_position = [1, 1]
        self.ghost_positions = [[GRID_SIZE - 2, GRID_SIZE - 2]]
        self.score = 0
        self.done = False

    def create_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        grid[0, :] = 1
        grid[:, 0] = 1
        grid[GRID_SIZE - 1, :] = 1
        grid[:, GRID_SIZE - 1] = 1
        for _ in range(100):
            x, y = random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)
            grid[x, y] = 2  # Pellet
        for _ in range(10):
            x, y = random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)
            grid[x, y] = 3  # Super Pellet
        return grid

    def reset(self):
        self.grid = self.create_grid()
        self.pacman_position = [1, 1]
        self.ghost_positions = [[GRID_SIZE - 2, GRID_SIZE - 2]]
        self.score = 0
        self.done = False
        return self.grid_state()

    def grid_state(self):
        state = self.grid.copy()
        state[self.pacman_position[0], self.pacman_position[1]] = 4  # Pac-Man
        for ghost in self.ghost_positions:
            state[ghost[0], ghost[1]] = 5  # Ghost
        return state.flatten()

    def step(self, action):
        if self.done:
            raise ValueError("Game is over. Call reset to start again.")

        # Move Pac-Man
        move = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dx, dy = move[action]
        nx, ny = self.pacman_position[0] + dx, self.pacman_position[1] + dy

        if self.grid[nx, ny] != 1:  # Check for walls
            self.pacman_position = [nx, ny]

        # Rewards and penalties (all cast to integers)
        reward = int(REWARD_STEP)
        if self.grid[nx, ny] == 2:  # Pellet
            reward += int(REWARD_PELLET)
            self.grid[nx, ny] = 0
        elif self.grid[nx, ny] == 3:  # Super Pellet
            reward += int(REWARD_SUPER_PELLET)
            self.grid[nx, ny] = 0

        # Ghost collision
        if any(self.pacman_position == ghost for ghost in self.ghost_positions):
            reward += int(REWARD_GHOST)
            self.done = True

        self.score += reward

        # Move Ghosts
        self.move_ghosts()

        # Check for game over
        if self.score < -500:
            self.done = True

        return self.grid_state(), reward, self.done, {}

    def move_ghosts(self):
        for ghost in self.ghost_positions:
            move = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            gx, gy = ghost[0] + move[0], ghost[1] + move[1]
            if self.grid[gx, gy] != 1:  # Avoid walls
                ghost[0], ghost[1] = gx, gy

    def render(self, screen):
        screen.fill((0, 0, 0))  # Clear the screen

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[x, y] == 1:  # Wall
                    pygame.draw.rect(screen, (0, 0, 255), pygame.Rect(y * TILE_SIZE, x * TILE_SIZE, TILE_SIZE, TILE_SIZE))
                elif self.grid[x, y] == 2:  # Pellet
                    pygame.draw.circle(screen, (255, 255, 0), (y * TILE_SIZE + TILE_SIZE // 2, x * TILE_SIZE + TILE_SIZE // 2), 4)
                elif self.grid[x, y] == 3:  # Super Pellet
                    pygame.draw.circle(screen, (255, 0, 255), (y * TILE_SIZE + TILE_SIZE // 2, x * TILE_SIZE + TILE_SIZE // 2), 6)

        pygame.draw.circle(screen, (255, 255, 0), (self.pacman_position[1] * TILE_SIZE + TILE_SIZE // 2, self.pacman_position[0] * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2)

        for ghost in self.ghost_positions:
            pygame.draw.circle(screen, (255, 0, 0), (ghost[1] * TILE_SIZE + TILE_SIZE // 2, ghost[0] * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 2)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))


        pygame.display.flip()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_pacman(env, episodes=500):
    state_dim = GRID_SIZE * GRID_SIZE
    action_dim = 4

    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    memory = deque(maxlen=2000)
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.1

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                q_values = model(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) > 32:
                batch = random.sample(memory, 32)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_q_values = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {int(total_reward)}")

    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.show()

    return model

if __name__ == "__main__":
    env = PacManEnv()
    
    mode = input("Enter 'play' to play the game manually, or 'watch' to train and watch the RL agent: ").strip().lower()
    
    if mode == 'play':
        print("Use arrow keys to move Pac-Man. Close the window to exit.")
        running = True
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Pac-Man")
        
        while running:
            env.reset()
            while not env.done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            env.step(0)
                        elif event.key == pygame.K_DOWN:
                            env.step(1)
                        elif event.key == pygame.K_LEFT:
                            env.step(2)
                        elif event.key == pygame.K_RIGHT:
                            env.step(3)

                if not running:
                    break

                env.render(screen)
                clock.tick(30)

            if running:
                play_again = input("Game over. Do you want to play again? (yes/no): ").strip().lower()
                if play_again != 'yes':
                    running = False

    elif mode == 'watch':
        episodes_to_train = int(input("Enter the number of episodes to train the RL agent: "))
        trained_model = train_pacman(env, episodes=episodes_to_train)
        torch.save(trained_model.state_dict(), "pacman_rl_model.pth")
        print("Training complete. Model saved as 'pacman_rl_model.pth'.")

        watch = input("Do you want to watch the RL agent play? (yes/no): ").strip().lower()
        if watch == 'yes':
            try:
                model = DQN(GRID_SIZE * GRID_SIZE, 4)
                model.load_state_dict(torch.load("pacman_rl_model.pth", weights_only=True))
                model.eval()
                print("Model loaded successfully. Starting the game...")
            except FileNotFoundError:
                print("Error: No trained model found. Please train the RL agent first.")
                exit()

            running = True
            clock = pygame.time.Clock()
            screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Pac-Man (RL Agent)")
            
            while running:
                env.reset()
                while not env.done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                    
                    if not running:
                        break

                    state = torch.FloatTensor(env.grid_state()).unsqueeze(0)
                    with torch.no_grad():
                        q_values = model(state)
                    action = torch.argmax(q_values).item()
                    env.step(action)

                    env.render(screen)
                    clock.tick(30)

                if running:
                    watch_again = input("Game over. Do you want to watch the RL agent play again? (yes/no): ").strip().lower()
                    if watch_again != 'yes':
                        running = False

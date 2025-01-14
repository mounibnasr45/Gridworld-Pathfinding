import numpy as np
import random
import tkinter as tk
from typing import Tuple, List
import time

class GridWorldGUI(tk.Tk):
    def __init__(self, size=20):
        super().__init__()
        
        self.size = size
        self.cell_size = 30
        self.canvas_size = self.cell_size * size
        
        self.title("Q-Learning GridWorld")
        
        # Create canvas
        self.canvas = tk.Canvas(self, width=self.canvas_size, 
                              height=self.canvas_size, bg='white')
        self.canvas.pack(side=tk.LEFT)
        
        # Control panel
        self.control_panel = tk.Frame(self)
        self.control_panel.pack(side=tk.RIGHT, padx=10)
        
        # Episode counter
        self.episode_label = tk.Label(self.control_panel, text="Episode: 0")
        self.episode_label.pack()
        
        # Reward display
        self.reward_label = tk.Label(self.control_panel, text="Total Reward: 0")
        self.reward_label.pack()
        
        # Epsilon display
        self.epsilon_label = tk.Label(self.control_panel, text="Epsilon: 1.0")
        self.epsilon_label.pack()
        
        # Start button
        self.start_button = tk.Button(self.control_panel, text="Start Training",
                                    command=self.start_training)
        self.start_button.pack()
        
        # Speed control
        self.speed_scale = tk.Scale(self.control_panel, from_=1, to=100,
                                  orient=tk.HORIZONTAL, label="Speed")
        self.speed_scale.set(50)
        self.speed_scale.pack()
        
        self.env = None
        self.agent = None
        self.training = False
        
    def draw_grid(self):
        self.canvas.delete("all")
        
        # Draw grid lines
        for i in range(self.size + 1):
            self.canvas.create_line(i * self.cell_size, 0, 
                                  i * self.cell_size, self.canvas_size)
            self.canvas.create_line(0, i * self.cell_size, 
                                  self.canvas_size, i * self.cell_size)
        
        # Draw obstacles
        for obs in self.env.obstacles:
            x, y = obs
            self.canvas.create_rectangle(y * self.cell_size, x * self.cell_size,
                                       (y + 1) * self.cell_size, 
                                       (x + 1) * self.cell_size, fill='black')
        
        # Draw goal
        x, y = self.env.goal
        self.canvas.create_rectangle(y * self.cell_size, x * self.cell_size,
                                   (y + 1) * self.cell_size, 
                                   (x + 1) * self.cell_size, fill='red')
        
        # Draw agent
        x, y = self.env.agent_pos
        self.canvas.create_oval(y * self.cell_size + 2, x * self.cell_size + 2,
                              (y + 1) * self.cell_size - 2, 
                              (x + 1) * self.cell_size - 2, fill='green')
        
        # Draw path
        if hasattr(self, 'current_path'):
            for i in range(len(self.current_path) - 1):
                x1, y1 = self.current_path[i]
                x2, y2 = self.current_path[i + 1]
                self.canvas.create_line((y1 + 0.5) * self.cell_size, 
                                     (x1 + 0.5) * self.cell_size,
                                     (y2 + 0.5) * self.cell_size, 
                                     (x2 + 0.5) * self.cell_size,
                                     fill='blue', width=2)
        
        self.update()

    def start_training(self):
        if not self.training:
            self.training = True
            self.start_button.config(text="Stop Training")
            self.env = GridWorld(self.size)
            self.agent = QLearningAgent(self.size, len(self.env.actions))
            self.train()
        else:
            self.training = False
            self.start_button.config(text="Start Training")

    def train(self):
        if not self.training:
            return
            
        state = self.env.reset()
        total_reward = 0
        self.current_path = [state]
        
        done = False
        while not done and self.training:
            action = self.agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            self.current_path.append(state)
            
            # Update GUI
            self.draw_grid()
            self.episode_label.config(text=f"Episode: {self.agent.episode}")
            self.reward_label.config(text=f"Total Reward: {total_reward}")
            self.epsilon_label.config(text=f"Epsilon: {self.agent.epsilon:.2f}")
            
            # Control speed
            delay = 100 - self.speed_scale.get()
            self.after(delay)
            
        self.agent.episode += 1
        if self.training:
            self.after(100, self.train)

class GridWorld:
    def __init__(self, size: int = 20):
        self.size = size
        self.obstacles = self.generate_obstacles()
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.agent_pos = self.start
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
    def generate_obstacles(self):
        obstacles = set()
        num_obstacles = self.size * self.size // 5  # 20% of grid cells are obstacles
        while len(obstacles) < num_obstacles:
            x = random.randint(0, self.size-1)
            y = random.randint(0, self.size-1)
            if (x, y) != (0, 0) and (x, y) != (self.size-1, self.size-1):
                obstacles.add((x, y))
        return list(obstacles)
    
    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos
    
    def step(self, action):
        move = self.actions[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            new_pos not in self.obstacles):
            self.agent_pos = new_pos
            
        if self.agent_pos == self.goal:
            reward = 100
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -50
            done = True
        else:
            reward = -1
            done = False
            
        return self.agent_pos, reward, done

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size
        self.episode = 0
        
    def get_action(self, state: Tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state[0]][state[1]])
    
    def learn(self, state: Tuple[int, int], action: int, 
              reward: float, next_state: Tuple[int, int]):
        old_value = self.q_table[state[0]][state[1]][action]
        next_max = np.max(self.q_table[next_state[0]][next_state[1]])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state[0]][state[1]][action] = new_value
        
        self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    app = GridWorldGUI()
    app.mainloop()
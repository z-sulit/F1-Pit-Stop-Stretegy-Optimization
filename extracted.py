%pip install seaborn

%pip install fastf1

%pip install scikit-learn

import fastf1
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
#--
import fastf1

fastf1.Cache.enable_cache('f1_cache')

#2
session = fastf1.get_session(2023, 'Monza', 'R')
session.load()

laps = session.laps
laps.head()
laps.columns


# Import the modules
from preprocessing import preprocess_laps
from validate_preprocessing import validate_preprocessing, plot_preprocessing_diagnostics
# Save original data for comparison (optional but recommended)
original_laps = laps.copy()
# Run preprocessing
laps = preprocess_laps(laps)

# Validate the preprocessed data
validation_results = validate_preprocessing(laps, original_laps)

# Show diagnostic plots
plot_preprocessing_diagnostics(laps)

print(laps.head())

print(laps[['Driver', 'LapNumber', 'LapTime', 'TyreLife', 'Compound']].head(40))

'''
Calculating Pitloss
Here we compare green flag lap time vs the Pit in and out lap times
'''
# Filter for drivers who finished, doing this removes crash data (drivers who crashed during the race)
clean_laps = laps.pick_quicklaps() # Removes slow laps ( like Yellow flags)
avg_race_pace = clean_laps['LapTimeSeconds'].median()

#Identify Pit Laps
pit_laps = laps[laps['PitOutTime'].notna()] # Laps where they exited the pits
avg_pit_lap_time = pit_laps['LapTimeSeconds'].median()

# Estimating PIt loss
pit_loss = avg_pit_lap_time - avg_race_pace

print(f"Avg Race Pace: {avg_race_pace} seconds")

print(f"Avg Pit lap Time: {avg_pit_lap_time} seconds")

print(f"Estimated Pit loss: {pit_loss} seconds")

#Making a tire degradation function
#Rival bot, Later, the goal is to beat this bot.

import random

class RaceParameters:
    base_lap_time = 87
    pit_loss = 21.00
def get_tire_degradation(tire_age, compound='medium'):
    if compound == 'soft':
        deg = 0.15 * tire_age
        if tire_age > 15: deg += 0.5 * (tire_age - 15)
    elif compound == 'medium':
        deg = 0.08 * tire_age
        if tire_age > 25: deg += 0.4 * (tire_age - 25)
    elif compound == 'hard':
        deg = 0.05 * tire_age
        if tire_age > 35: deg += 0.3 * (tire_age - 35)
    else:
        raise ValueError(f"Unknown compound: {compound}")
    return deg

class RivalBot:
    def __init__(self, pit_lap=24, start_tire='medium', switch_tire='hard'):
        self.pit_lap = pit_lap
        self.current_tire = start_tire
        self.switch_tire = switch_tire
        self.tire_age = 0

    def get_action(self, current_lap):
        if current_lap == self.pit_lap:
            self.current_tire = self.switch_tire
            self.tire_age = 0
            return 1  # Pit
        
        self.tire_age += 1
        return 0  # Stay Out


%pip install gymnasium

import gymnasium as gym
from gymnasium import spaces

class F1PitStopEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self):
        super(F1PitStopEnv, self).__init__()
        
        # Action Space: 
        # 0 = Stay out, 1 = Pit Soft, 2 = Pit Medium, 3 = Pit Hard
        self.action_space = spaces.Discrete(4)
        
        # State Space Box (Normalized [0, 1]):
        # [Current Lap, Tire Age, Compound, Gap to Rival]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.total_laps = 51
        self.pit_loss_seconds = 20.30
        
        # --- Internal State Variables ---
        self.current_lap = 0
        self.current_tire_age = 0
        self.current_compound = 0  # 0=Soft, 1=Medium, 2=Hard
        self.gap_to_rival = 0.0    # Seconds
        
        # Race Management
        self.agent_total_time = 0.0
        self.num_stops = 0
        self.compounds_used = set()
        
        # Rival Placeholder
        self.rival_bot = None 
        self.rival_total_time = 0.0 # Will be populated by rival bot

    def _get_obs(self):
        """Helper to normalize the state into the [0, 1] observation space."""
        # 1. Normalize Lap & Tire Age
        norm_lap = self.current_lap / self.total_laps
        norm_age = self.current_tire_age / self.total_laps
        
        # 2. Normalize Compound (0, 1, 2 becomes 0.0, 0.5, 1.0)
        norm_compound = self.current_compound / 2.0
        
        # 3. Normalize Gap (Shift [-120s, +120s] -> [0.0, 1.0])
        norm_gap = np.clip((self.gap_to_rival + 120) / 240, 0.0, 1.0)
        
        return np.array([norm_lap, norm_age, norm_compound, norm_gap], dtype=np.float32)

    def _calculate_lap_time(self):
        """
        Calculates lap time based on compound and age.
        Placeholder for the Phase 1 Tire Degradation Model.
        """
        # Base lap time (e.g., Monza avg)
        base_lap_time = 86.5  

        # Degradation rates (seconds lost per lap of age)
        deg_rates = {
            0: 0.10,  # Soft  - Fast deg
            1: 0.06,  # Medium
            2: 0.04,  # Hard  - Slow deg
        }
        
        # Tire wear penalty
        degradation = deg_rates[self.current_compound] * self.current_tire_age
        
        return base_lap_time + degradation

    def _get_rival_time_at_lap(self, lap):
        """
        Rival strategy: Start on Mediums, pit on Lap 20 for Hards.
        Same degradation model as the agent.
        """
        base = 86.5
        total = 0.0
        for i in range(lap):
            if i < 20:
            # Stint 1: Mediums (deg 0.06s/lap)
                total += base + 0.06 * i
            elif i == 20:
            # Pit lap: fresh Hards + pit loss
                total += base + 0.0 + self.pit_loss_seconds  # tire_age resets to 0
            else:
            # Stint 2: Hards (deg 0.04s/lap), age = laps since pit
                tire_age = i - 20
                total += base + 0.04 * tire_age
        return total

    def step(self, action):
        """
        Executes one lap of the race based on the agent's action.
        """
        terminated = False
        truncated = False
        info = {}
        reward = 0.0

        is_pitting = False

        if action == 0:
            # Stay out
            pass
        elif action in [1, 2, 3]:
            # Pit stop
            is_pitting = True
            new_compound = action - 1  # 0->Soft, 1->Med, 2->Hard
            
            self.current_compound = new_compound
            self.compounds_used.add(new_compound)
            self.current_tire_age = 0  # Reset age on new tires
            self.num_stops += 1


        #CALCULATE LAP TIME

        lap_time = self._calculate_lap_time()

        # Add pit loss if we pitted
        if is_pitting:
            lap_time += self.pit_loss_seconds

        # Update Agent's Race Time
        self.agent_total_time += lap_time

        #STEP REWARD

        # Penalize time taken (Agent wants to minimize this negative number)
        reward = -lap_time

        #ADVANCE STATE

        self.current_lap += 1
        self.current_tire_age += 1

        # Update Gap (Positive = Ahead, Negative = Behind)
        # We need the rival's time at the *end* of this lap
        rival_cumulative_time = self._get_rival_time_at_lap(self.current_lap)
        self.gap_to_rival = rival_cumulative_time - self.agent_total_time


        #CHECK TERMINAL CONDITIONS
    
        if self.current_lap >= self.total_laps:
            terminated = True
            
            # Retrieve rival's total time (if not set, use current projection)
            final_rival_time = self.rival_total_time if self.rival_total_time > 0 else rival_cumulative_time

            # A. Invalid Action Check (Must use 2+ compounds)
            if len(self.compounds_used) < 2:
                reward += -500 
                info['outcome'] = 'Disqualified (1 Compound)'
            
            # B. Win/Loss Check
            elif self.agent_total_time < final_rival_time:
                reward += 100
                info['outcome'] = 'Win'
            else:
                reward += -100
                info['outcome'] = 'Loss'

        observation = self._get_obs()
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        self.current_lap = 0
        self.current_tire_age = 0
        self.current_compound = 0
        self.gap_to_rival = 0.0
        self.agent_total_time = 0.0
        self.num_stops = 0
        self.compounds_used = {self.current_compound}
        # Pre-calculate rival's total race time
        self.rival_total_time = self._get_rival_time_at_lap(self.total_laps)
    
        return self._get_obs(), {}

    def render(self):
        print(f"Lap: {self.current_lap} | Gap: {self.gap_to_rival:.2f}s | Action: {self.current_compound}")

class QLearningAgent:
    def __init__(self, env, bins_per_feature=10,
                 learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env = env
        self.bins_per_feature = bins_per_feature
        self.bins = [
            np.linspace(0, 1, bins_per_feature + 1)[1:-1]
            for _ in range(4)
        ]
        q_shape = tuple([bins_per_feature] * 4 + [env.action_space.n])
        self.q_table = np.zeros(q_shape)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def _discretize(self, obs):
        state = []
        for i, val in enumerate(obs):
            bin_index = np.digitize(val, self.bins[i])
            state.append(bin_index)
        return tuple(state)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state + (action,)]
        if terminated:
            target = reward
        else:
            best_future_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_future_q
        self.q_table[state + (action,)] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)



#If bins_per_feature=10, each feature gets split into 10 buckets. So "lap 0.25" goes into bin 2, "gap 0.5" goes into bin 5, etc. Total Q-Table size = 10 × 10 × 10 × 10 × 4 = 40,000 entries......


def train(env, agent, num_episodes=5000):
    rewards_history = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = agent._discretize(obs)
        total_reward = 0
        terminated = False
        while not terminated:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent._discretize(next_obs)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        if (episode + 1) % 500 == 0:
            avg = np.mean(rewards_history[-500:])
            print(f"Episode {episode+1} | Avg Reward: {avg:.1f} | Epsilon: {agent.epsilon:.3f}")
    return rewards_history

def run_test_episodes(env, agent, num_episodes=100):
    results = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        state = agent._discretize(obs)
        terminated = False
        total_reward = 0
        pit_laps = []
        compounds_sequence = ['Soft']
        lap = 0
        while not terminated:
            action = np.argmax(agent.q_table[state])
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent._discretize(next_obs)
            if action in [1, 2, 3]:
                pit_laps.append(lap)
                compounds_sequence.append(['Soft', 'Medium', 'Hard'][action - 1])
            state = next_state
            total_reward += reward
            lap += 1
        results.append({
            'total_reward': total_reward,
            'pit_laps': pit_laps,
            'compounds_sequence': compounds_sequence,
            'outcome': info.get('outcome', 'Unknown'),
            'agent_time': env.agent_total_time,
            'rival_time': env.rival_total_time if env.rival_total_time > 0
                          else env._get_rival_time_at_lap(env.total_laps)
        })
    return results


env = F1PitStopEnv()
agent = QLearningAgent(env, bins_per_feature=20, epsilon_decay=0.9995)
rewards = train(env, agent, num_episodes=20000)



import matplotlib.pyplot as plt
from collections import Counter


print(type(env))
print(type(agent))
print(len(rewards))

results = run_test_episodes(env, agent, num_episodes=100)
RIVAL_PIT_LAP = 20  # The rival's known pit lap


print(f"Rival pits on Lap {RIVAL_PIT_LAP}. Agent should pit BEFORE that.\n")

first_pit_laps = [r['pit_laps'][0] for r in results if len(r['pit_laps']) > 0]

if first_pit_laps:
    pit_counts = Counter(first_pit_laps)
    mode_lap = pit_counts.most_common(1)[0][0]
    avg_lap = np.mean(first_pit_laps)

    print(f"Mode (most frequent) pit lap: {mode_lap}")
    print(f"Average first pit lap:        {avg_lap:.1f}")
    print(f"Distribution: {dict(sorted(pit_counts.items()))}")

    if mode_lap < RIVAL_PIT_LAP:
        print("\n✅ PASS — Agent undercuts the rival!")
    else:
        print(f"\n❌ FAIL — Mode pit lap ({mode_lap}) is NOT before rival ({RIVAL_PIT_LAP})")

    # Over-eager check
    if mode_lap < 15:
        print(f"!!!! Agent pits very early (lap {mode_lap}).")
        print("Risk of running out of tire life at the end of the race.")
else:
    print("fAIL, the agent never pitted")



print("The -500 penalty must act as an 'electric fence'.\n")

dq_count = sum(1 for r in results if 'Disqualified' in r['outcome'])
dq_rate = dq_count / len(results) * 100

print(f"Disqualifications: {dq_count}/{len(results)} ({dq_rate:.0f}%)")

if dq_count == 0:
    print("\nPASS — 0% disqualification rate")
else:
    print(f"\nFAIL — {dq_rate:.0f}% disqualification rate")
    print("If this persists after 5,000+ episodes, increase")
    print("the penalty from -500 to -1000, or verify compounds_used")
    print("state is being correctly tracked.")


print("Agent should avoid Soft→Soft (same compound) and find a")
print("sensible sequence like S→M, S→H, or M→H.\n")

strategy_counts = Counter()
for r in results:
    seq = ' → '.join(r['compounds_sequence'])
    strategy_counts[seq] += 1

print("Strategy frequency (top 5):")
for strategy, count in strategy_counts.most_common(5):
    pct = count / len(results) * 100
    print(f"  {strategy}: {count}x ({pct:.0f}%)")

# Check for illegal same-compound transitions
illegal_count = 0
for r in results:
    seq = r['compounds_sequence']
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            illegal_count += 1
            break

if illegal_count == 0:
    print("\nNo same-compound transitions detected")
else:
    print(f"\n{illegal_count} races had same-compound transitions")

# Note about S→H
top_strategy = strategy_counts.most_common(1)[0][0] if strategy_counts else ""
if 'Hard' in top_strategy and 'Medium' not in top_strategy:
    print("ℹAgent prefers S→H: prioritizing durability over pace for stint 2")


fig, ax = plt.subplots(figsize=(12, 5))

# Raw rewards (faded)
ax.plot(rewards, alpha=0.2, color='blue', label='Per Episode')

# Moving average (bold)
window = 100
if len(rewards) >= window:
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(rewards)), moving_avg, color='red',
            linewidth=2, label=f'{window}-Episode Moving Avg')

ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Q-Learning Training Progress')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


early_avg = np.mean(rewards[:500])
late_avg = np.mean(rewards[-500:])
mid_avg = np.mean(rewards[len(rewards)//2 - 250 : len(rewards)//2 + 250])

print(f"\nEarly avg reward (first 500):  {early_avg:.1f}")
print(f"Mid avg reward (middle 500):   {mid_avg:.1f}")
print(f"Late avg reward (last 500):    {late_avg:.1f}")

# Check if still climbing or plateaued
if late_avg > mid_avg + 5:
    print("\nCurve is STILL climbing — agent hasn't finished learning.")
    print("   Consider training for more episodes (e.g., 10,000).")
elif late_avg > early_avg:
    print("\nPASS — Curve has plateaued. Training is stable.")
else:
    print("\nFAIL — No improvement detected. Check hyperparameters.")



wins = sum(1 for r in results if r['outcome'] == 'Win')
losses = sum(1 for r in results if r['outcome'] == 'Loss')
dqs = sum(1 for r in results if 'Disqualified' in r['outcome'])
win_rate = wins / len(results) * 100

print(f"\nWins: {wins} | Losses: {losses} | DQs: {dqs}")
print(f"Win Rate: {win_rate:.0f}%")

if win_rate == 100:
    print("\nPASS — 100% win rate!")
    print("WARNING: Rival might be too slow (87.0s is conservative).")
    print('   For "Hard Mode", lower rival lap time to 86.8s and re-test.')
elif win_rate >= 80:
    print(f"\nPASS — {win_rate:.0f}% win rate (≥80% threshold)")
elif win_rate >= 50:
    print(f"\nPARTIAL — {win_rate:.0f}% win rate (below 80%)")
else:
    print(f"\nFAIL — {win_rate:.0f}% win rate. Agent loses more than it wins.")



print("\n" + "=" * 60)
print("FINAL PHASE 4 SUMMARY")
avg_agent = np.mean([r['agent_time'] for r in results])
avg_rival = np.mean([r['rival_time'] for r in results])
print(f"Avg Agent Race Time:  {avg_agent:.1f}s")
print(f"Avg Rival Race Time:  {avg_rival:.1f}s")
print(f"Avg Time Advantage:   {avg_rival - avg_agent:+.1f}s")
print(f"Win Rate:             {win_rate:.0f}%")
print(f"Disqualification Rate: {dq_rate:.0f}%")
if first_pit_laps:
    print(f"Mode Pit Lap:         {mode_lap} (Rival: {RIVAL_PIT_LAP})")



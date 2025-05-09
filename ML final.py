import gym
import gymnasium as gym_new  # For mujoco environment (SAC)
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from functools import lru_cache

# ===========================================================================
# 1. Global Configuration and Utility Functions
# ===========================================================================
np.random.seed(0)

# Discount factors, learning rate, epsilon parameters, number of episodes, moving average windows
GAMMA_BLACKJACK = 1.0
GAMMA_CARTPOLE = 0.99
ALPHA = 0.1
EPSILON_START = 1.0
EPSILON_END = 0.1

EPISODES_BLACKJACK = 50000
EPISODES_CARTPOLE = 5000

MA_WINDOW_BLACKJACK = 1000
MA_WINDOW_CARTPOLE = 100

# 결과 이미지 및 모델 저장 경로 (컴퓨터에 맞게 수정)
OUTPUT_DIR = r"C:/Users/slee/OneDrive - SBP/Desktop/sungkeun/omscs/Spring2025/CS7641/Assignment/A4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(filename):
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved {filename} to {OUTPUT_DIR}")

def moving_average(data, window=100):
    """Compute the moving average with the given window."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ===========================================================================
# 2. Environment Setup
# ===========================================================================
# Blackjack: try to use v1; if not available, fallback to v0.
try:
    env_blackjack = gym.make('Blackjack-v1')
except Exception:
    env_blackjack = gym.make('Blackjack-v0')
# CartPole environment
env_cartpole = gym.make('CartPole-v1')

# Reset environments with a seed for reproducibility
env_blackjack.reset(seed=0)
env_cartpole.reset(seed=0)

# ===========================================================================
# 3. CartPole Discretization Utility
# ===========================================================================
def discretize_cartpole(obs, bins=(10, 10, 10, 10)):
    """
    Discretize a CartPole observation (x, x_dot, theta, theta_dot) 
    into a tuple of bin indices.
    """
    x, x_dot, theta, theta_dot = obs
    pos_bins  = np.linspace(-4.8, 4.8, bins[0]+1)
    vel_bins  = np.linspace(-3.0, 3.0, bins[1]+1)
    theta_bins = np.linspace(-0.418, 0.418, bins[2]+1)  # ~24° in radians
    ang_vel_bins = np.linspace(-4.0, 4.0, bins[3]+1)
    x_c     = np.clip(x, -4.8, 4.8)
    x_dot_c = np.clip(x_dot, -3.0, 3.0)
    theta_c = np.clip(theta, -0.418, 0.418)
    theta_dot_c = np.clip(theta_dot, -4.0, 4.0)
    i0 = int(np.digitize(x_c, pos_bins) - 1);    i0 = min(i0, bins[0]-1)
    i1 = int(np.digitize(x_dot_c, vel_bins) - 1); i1 = min(i1, bins[1]-1)
    i2 = int(np.digitize(theta_c, theta_bins) - 1); i2 = min(i2, bins[2]-1)
    i3 = int(np.digitize(theta_dot_c, ang_vel_bins) - 1); i3 = min(i3, bins[3]-1)
    return (i0, i1, i2, i3)

# ===========================================================================
# 4. Blackjack MDP Model and DP Solutions
# ===========================================================================
# Define state space: (player_sum 4-21, dealer showing 1-10, usable ace True/False)
states_blackjack = []
for total in range(4, 22):
    for dealer in range(1, 11):
        for usable in [False, True]:
            states_blackjack.append((total, dealer, usable))
num_states_bj = len(states_blackjack)
actions_bj = [0, 1]  # 0 = stick, 1 = hit
state_index_bj = {s: i for i, s in enumerate(states_blackjack)}

# Infinite deck rules
deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]

def sum_hand(hand):
    total = sum(hand)
    if 1 in hand and total + 10 <= 21:
        return total + 10
    return total

def is_usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

@lru_cache(maxsize=None)
def dealer_outcome_probs(dealer_hand_total, dealer_has_usable_ace):
    """
    Given the dealer's current total and usable ace status,
    return a dictionary of final outcomes (17-21 or 'bust') and probabilities.
    """
    if dealer_hand_total >= 17:
        return {'bust': 1.0} if dealer_hand_total > 21 else {dealer_hand_total: 1.0}
    outcomes = defaultdict(float)
    for card in deck:
        prob = 1.0/len(deck)
        total_base = dealer_hand_total - (10 if dealer_has_usable_ace else 0)
        new_total_base = total_base + (1 if card == 1 else card)
        has_ace = (1 if dealer_has_usable_ace else 0) + (1 if card == 1 else 0)
        if has_ace > 0 and new_total_base + 10 <= 21:
            new_total = new_total_base + 10
            new_usable = True
        else:
            new_total = new_total_base
            new_usable = False
        for outcome, p in dealer_outcome_probs(new_total, new_usable).items():
            outcomes[outcome] += prob * p
    return outcomes

def dealer_initial_state(upcard, hole):
    cards = [upcard, hole]
    total = sum(cards)
    return (total + 10, True) if 1 in cards and total + 10 <= 21 else (total, False)

def expected_stick_reward(player_sum, dealer_upcard):
    exp_reward = 0.0
    dealer_dist = {}
    for hole in deck:
        prob = 1.0/len(deck)
        init_sum, usable = dealer_initial_state(dealer_upcard, hole)
        for outcome, p in dealer_outcome_probs(init_sum, usable).items():
            dealer_dist[outcome] = dealer_dist.get(outcome, 0) + prob * p
    for outcome, p in dealer_dist.items():
        reward = 1.0 if outcome == 'bust' else (1.0 if outcome < player_sum else (-1.0 if outcome > player_sum else 0.0))
        exp_reward += p * reward
    return exp_reward

def next_player_state(player_sum, player_usable, card):
    if card == 1:
        if player_sum + 11 <= 21:
            new_sum, new_usable = player_sum + 11, True
        else:
            new_sum, new_usable = player_sum + 1, player_usable
            if new_sum > 21 and player_usable:
                new_sum, new_usable = new_sum - 10, False
    else:
        new_sum, new_usable = player_sum + card, player_usable
        if new_sum > 21 and player_usable:
            new_sum, new_usable = new_sum - 10, False
    if new_sum > 21:
        return None, None, True, -1.0
    return new_sum, new_usable, False, 0.0

def value_iteration_blackjack(gamma=1.0, theta=1e-6):
    # Only consider valid states
    states = [s for s in states_blackjack if not (s[2] and s[0] < 12)]
    V = {s: 0.0 for s in states}
    policy = {s: 0 for s in states}
    deltas = []
    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        for s in states:
            ps, d, u = s
            Q_stick = expected_stick_reward(ps, d)
            Q_hit = 0.0
            for card in deck:
                prob = 1.0/len(deck)
                new_sum, new_usable, done, reward = next_player_state(ps, u, card)
                if done:
                    Q_hit += prob * reward
                else:
                    next_state = (new_sum, d, new_usable)
                    Q_hit += prob * (reward + gamma * V[next_state])
            best_val = max(Q_stick, Q_hit)
            delta = max(delta, abs(best_val - V[s]))
            V[s] = best_val
        deltas.append(delta)
        if delta < theta:
            for s in states:
                ps, d, u = s
                Q_stick = expected_stick_reward(ps, d)
                Q_hit = 0.0
                for card in deck:
                    prob = 1.0/len(deck)
                    new_sum, new_usable, done, reward = next_player_state(ps, u, card)
                    if done:
                        Q_hit += prob * reward
                    else:
                        next_state = (new_sum, d, new_usable)
                        Q_hit += prob * (reward + gamma * V[next_state])
                policy[s] = 0 if Q_stick >= Q_hit else 1
            break
    print(f"Blackjack Value Iteration converged in {iteration} iterations (policy stable at iter {iteration}).")
    return V, policy, deltas, iteration

def policy_iteration_blackjack(gamma=1.0, theta=1e-6):
    states = [s for s in states_blackjack if not (s[2] and s[0] < 12)]
    policy = {s: (0 if s[0] >= 20 else 1) for s in states}
    V = {s: 0.0 for s in states}
    deltas = []
    iter_count = 0
    while True:
        iter_count += 1
        while True:
            delta = 0.0
            for s in states:
                ps, d, u = s
                a = policy[s]
                v = 0.0
                for card in deck:
                    prob = 1.0/len(deck)
                    new_sum, new_usable, done, reward = next_player_state(ps, u, card)
                    if done:
                        v += prob * reward
                    else:
                        next_state = (new_sum, d, new_usable)
                        v += prob * (reward + gamma * V[next_state])
                diff = abs(v - V[s])
                V[s] = v
                delta = max(delta, diff)
            if delta < theta:
                break
        policy_stable = True
        for s in states:
            ps, d, u = s
            old_a = policy[s]
            Q_stick = expected_stick_reward(ps, d)
            Q_hit = 0.0
            for card in deck:
                prob = 1.0/len(deck)
                new_sum, new_usable, done, reward = next_player_state(ps, u, card)
                if done:
                    Q_hit += prob * reward
                else:
                    next_state = (new_sum, d, new_usable)
                    Q_hit += prob * (reward + gamma * V[next_state])
            best_action = 0 if Q_stick >= Q_hit else 1
            if best_action != old_a:
                policy_stable = False
            policy[s] = best_action
        deltas.append(delta)
        print(f"Policy Iteration step {iter_count}: policy stable = {policy_stable}")
        if policy_stable:
            break
    print(f"Blackjack Policy Iteration converged in {iter_count} iterations.")
    return V, policy, deltas, iter_count

# -------------------------------------------------------------------
# 5. Model-Free Learning on Blackjack: Q-Learning vs SARSA
# -------------------------------------------------------------------
def train_blackjack_q_sarsa(num_episodes=50000, alpha=0.1, gamma=0.99,
                            epsilon=1.0, epsilon_min=0.01, decay_rate=0.9995, method="Q"):
    Q = {}
    def get_Q(state):
        if state not in Q:
            Q[state] = np.zeros(2)
        return Q[state]
    eps = epsilon
    reward_history = []
    for episode in range(1, num_episodes+1):
        state, _ = env_blackjack.reset()
        done = False
        if method == "SARSA":
            if np.random.rand() < eps:
                action = env_blackjack.action_space.sample()
            else:
                action = int(np.argmax(get_Q(state)))
        while not done:
            if method == "Q":
                if np.random.rand() < eps:
                    action = env_blackjack.action_space.sample()
                else:
                    action = int(np.argmax(get_Q(state)))
            next_state, reward, terminated, truncated, _ = env_blackjack.step(action)
            done = terminated or truncated
            if method == "SARSA":
                if not done:
                    if np.random.rand() < eps:
                        next_action = env_blackjack.action_space.sample()
                    else:
                        next_action = int(np.argmax(get_Q(next_state)))
                else:
                    next_action = None
            if method == "Q":
                target = reward if done else reward + gamma * np.max(get_Q(next_state))
                get_Q(state)[action] += alpha * (target - get_Q(state)[action])
            elif method == "SARSA":
                target = reward if done else reward + gamma * get_Q(next_state)[next_action]
                get_Q(state)[action] += alpha * (target - get_Q(state)[action])
            state = next_state
            if method == "SARSA":
                action = next_action
        if eps > epsilon_min:
            eps *= decay_rate
            if eps < epsilon_min:
                eps = epsilon_min
        reward_history.append(reward)
        if episode % 10000 == 0:
            avg_reward = np.mean(reward_history[-1000:])
            print(f"{method} Episode {episode}: recent avg reward = {avg_reward:.3f}, epsilon = {eps:.3f}")
    return Q, reward_history

Q_Qlearn_bj, rewards_Q_bj = train_blackjack_q_sarsa(num_episodes=50000, alpha=0.1, method="Q")
Q_SARSA_bj, rewards_S_bj = train_blackjack_q_sarsa(num_episodes=50000, alpha=0.1, method="SARSA")

# -------------------------------------------------------------------
# 6. CartPole: DP, Transition Model, and Policy Visualization
# -------------------------------------------------------------------
# Discretize CartPole with 10 bins per state dimension
num_bins = (10, 10, 10, 10)
N0, N1, N2, N3 = num_bins
num_states_cp = N0 * N1 * N2 * N3
actions_cp = [0, 1]

# Map discrete state index <-> tuple
cp_index_to_state = {}
cp_state_to_index = {}
idx = 0
for i in range(N0):
    for j in range(N1):
        for k in range(N2):
            for l in range(N3):
                cp_index_to_state[idx] = (i, j, k, l)
                cp_state_to_index[(i, j, k, l)] = idx
                idx += 1

# Precompute bin center values for each dimension
bin_edges = {
    0: np.linspace(-4.8, 4.8, N0+1),
    1: np.linspace(-3.0, 3.0, N1+1),
    2: np.linspace(-0.418, 0.418, N2+1),
    3: np.linspace(-4.0, 4.0, N3+1)
}
bin_centers = {}
for dim in range(4):
    edges = bin_edges[dim]
    bin_centers[dim] = (edges[:-1] + edges[1:]) / 2.0

# Build transition model for CartPole using Euler integration from bin centers
transitions_cp = {s_idx: {} for s_idx in range(num_states_cp)}
gravity = 9.8; masscart = 1.0; masspole = 0.1; total_mass = masscart + masspole
length = 0.5; polemass_length = masspole * length; force_mag = 10.0; tau = 0.02

for s_idx, (i, j, k, l) in cp_index_to_state.items():
    cont_state = (bin_centers[0][i], bin_centers[1][j], bin_centers[2][k], bin_centers[3][l])
    for action in actions_cp:
        x, x_dot, theta, theta_dot = cont_state
        force = force_mag if action == 1 else -force_mag
        cos_theta = math.cos(theta); sin_theta = math.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sin_theta) / total_mass
        thetaacc = (gravity * sin_theta - cos_theta * temp) / (length * (4.0/3.0 - masspole * cos_theta**2 / total_mass))
        xacc = temp - polemass_length * thetaacc * cos_theta / total_mass
        x_new = x + tau * x_dot
        x_dot_new = x_dot + tau * xacc
        theta_new = theta + tau * theta_dot
        theta_dot_new = theta_dot + tau * thetaacc
        done = (x_new < -4.8 or x_new > 4.8 or theta_new < -0.418 or theta_new > 0.418)
        reward = 1.0
        if done:
            transitions_cp[s_idx][action] = (None, reward, True)
        else:
            next_state_tuple = discretize_cartpole((x_new, x_dot_new, theta_new, theta_dot_new), bins=num_bins)
            next_idx = cp_state_to_index[next_state_tuple]
            transitions_cp[s_idx][action] = (next_idx, reward, False)

# Value Iteration for CartPole
gamma_cp = 0.99
theta_cp_val = 1e-3
V_cp = np.zeros(num_states_cp)
vi_cp_iterations = 0
while True:
    vi_cp_iterations += 1
    delta = 0.0
    for s_idx in range(num_states_cp):
        best_val = -float('inf')
        for a in actions_cp:
            next_info = transitions_cp[s_idx][a]
            next_state, reward, done = next_info
            q_val = reward + (0 if done else gamma_cp * V_cp[next_state])
            best_val = max(best_val, q_val)
        delta = max(delta, abs(best_val - V_cp[s_idx]))
        V_cp[s_idx] = best_val
    if delta < theta_cp_val:
        break
print(f"CartPole VI converged in {vi_cp_iterations} iterations (θ={theta_cp_val}).")

# Derive greedy policy for CartPole
policy_cp = np.zeros(num_states_cp, dtype=int)
for s_idx in range(num_states_cp):
    best_a = 0
    best_val = -float('inf')
    for a in actions_cp:
        next_info = transitions_cp[s_idx][a]
        next_state, reward, done = next_info
        q_val = reward + (0 if done else gamma_cp * V_cp[next_state])
        if q_val > best_val:
            best_val = q_val
            best_a = a
    policy_cp[s_idx] = best_a

# Visualize a policy slice: fix cart velocity and pole angular velocity at mid
mid_vel_idx = N1 // 2
mid_angvel_idx = N3 // 2
policy_slice = np.zeros((N0, N2), dtype=int)
for i in range(N0):
    for k in range(N2):
        s_idx = cp_state_to_index[(i, mid_vel_idx, k, mid_angvel_idx)]
        policy_slice[i, k] = policy_cp[s_idx]
plt.figure(figsize=(6,5))
plt.imshow(policy_slice, origin='lower', cmap='Paired')
plt.colorbar(ticks=[0,1], label='Action (0=Left, 1=Right)')
plt.xlabel('Pole Angle Index')
plt.ylabel('Cart Position Index')
plt.title('CartPole Policy Slice (Velocities ~0)')
plt.tight_layout()
save_fig("CartPole_policy_slice.png")
plt.show()

# -------------------------------------------------------------------
# 7. Model-Free Learning on CartPole: Q-Learning, SARSA, and Double Q-Learning
# -------------------------------------------------------------------
def train_cartpole(agent="Q", num_episodes=10000, alpha=0.5, gamma=0.99,
                   epsilon=1.0, epsilon_min=0.01, decay_rate=0.995):
    if agent == "DoubleQ":
        Q1 = np.zeros((N0, N1, N2, N3, 2))
        Q2 = np.zeros((N0, N1, N2, N3, 2))
    else:
        Q = np.zeros((N0, N1, N2, N3, 2))
    reward_history = []
    eps = epsilon
    for episode in range(1, num_episodes+1):
        obs, _ = env_cartpole.reset()
        state = discretize_cartpole(obs, bins=num_bins)
        done = False
        if agent == "SARSA":
            if np.random.rand() < eps:
                action = env_cartpole.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))
        total_reward = 0
        while not done:
            if agent == "Q" or agent == "DoubleQ":
                if np.random.rand() < eps:
                    action = env_cartpole.action_space.sample()
                else:
                    if agent == "Q":
                        action = int(np.argmax(Q[state]))
                    else:
                        action = int(np.argmax(Q1[state] + Q2[state]))
            obs_next, reward, terminated, truncated, _ = env_cartpole.step(action)
            done = terminated or truncated
            next_state = discretize_cartpole(obs_next, bins=num_bins) if not done else None
            if agent == "SARSA":
                if not done:
                    if np.random.rand() < eps:
                        next_action = env_cartpole.action_space.sample()
                    else:
                        next_action = int(np.argmax(Q[next_state]))
                else:
                    next_action = None
            if agent == "Q":
                target = reward if done else reward + gamma * np.max(Q[next_state])
                Q[state][action] += alpha * (target - Q[state][action])
            elif agent == "SARSA":
                target = reward if done else reward + gamma * Q[next_state][next_action]
                Q[state][action] += alpha * (target - Q[state][action])
            elif agent == "DoubleQ":
                if np.random.rand() < 0.5:
                    if done:
                        target = reward
                    else:
                        a_max = int(np.argmax(Q1[next_state]))
                        target = reward + gamma * Q2[next_state][a_max]
                    Q1[state][action] += alpha * (target - Q1[state][action])
                else:
                    if done:
                        target = reward
                    else:
                        a_max = int(np.argmax(Q2[next_state]))
                        target = reward + gamma * Q1[next_state][a_max]
                    Q2[state][action] += alpha * (target - Q2[state][action])
            state = next_state
            if agent == "SARSA":
                action = next_action
            total_reward += reward
        reward_history.append(env_cartpole._elapsed_steps if hasattr(env_cartpole, "_elapsed_steps") else total_reward)
        if eps > epsilon_min:
            eps *= decay_rate
            if eps < epsilon_min:
                eps = epsilon_min
        if episode % 1000 == 0:
            avg_rew = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            print(f"{agent} Episode {episode}: last 100 episodes avg reward = {avg_rew:.1f}, epsilon = {eps:.3f}")
    if agent == "DoubleQ":
        return (Q1, Q2), reward_history
    else:
        return Q, reward_history

def compute_avg_curve(reward_hist, window=100):
    avg = []
    for i in range(len(reward_hist)):
        if i < window:
            avg.append(np.mean(reward_hist[:i+1]))
        else:
            avg.append(np.mean(reward_hist[i-window+1:i+1]))
    return avg

Q_cartpole_Q, rewards_Q_cp = train_cartpole(agent="Q", num_episodes=EPISODES_CARTPOLE, alpha=0.5)
Q_cartpole_S, rewards_S_cp = train_cartpole(agent="SARSA", num_episodes=EPISODES_CARTPOLE, alpha=0.5)
(Q_cartpole_D, rewards_D_cp) = train_cartpole(agent="DoubleQ", num_episodes=EPISODES_CARTPOLE, alpha=0.5)

avg_rewards_Q_cp = compute_avg_curve(rewards_Q_cp, window=MA_WINDOW_CARTPOLE)
avg_rewards_S_cp = compute_avg_curve(rewards_S_cp, window=MA_WINDOW_CARTPOLE)
avg_rewards_D_cp = compute_avg_curve(rewards_D_cp, window=MA_WINDOW_CARTPOLE)

plt.figure(figsize=(8,5))
plt.plot(avg_rewards_Q_cp, label='Q-Learning')
plt.plot(avg_rewards_S_cp, label='SARSA')
plt.plot(avg_rewards_D_cp, label='Double Q-Learning')
plt.axhline(195, color='grey', linestyle='--', label='Solved Threshold')
plt.xlabel('Episode')
plt.ylabel(f'Avg Reward (window={MA_WINDOW_CARTPOLE} episodes)')
plt.title('CartPole: Q-Learning vs SARSA vs Double Q-Learning')
plt.legend()
save_fig("CartPole_Q_SARSA_DoubleQ.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(avg_rewards_Q_cp, label='Q-Learning')
plt.plot(avg_rewards_S_cp, label='SARSA')
plt.axhline(195, color='grey', linestyle='--', label='Solved Threshold')
plt.xlabel('Episode')
plt.ylabel(f'Avg Reward (window={MA_WINDOW_CARTPOLE} episodes)')
plt.title('CartPole: SARSA vs Q-Learning')
plt.legend()
save_fig("CartPole_SARSA_vs_Q.png")
plt.show()

# ===========================================================================
# 7. Extra Credit: Inverted Double Pendulum with SAC
# ===========================================================================
try:
    env_pendulum = gym_new.make('InvertedDoublePendulum-v4')
except Exception as e:
    print("Failed to load InvertedDoublePendulum-v4. Ensure MuJoCo is installed.")
    env_pendulum = None

if env_pendulum is not None:
    from stable_baselines3 import SAC
    model_SAC = SAC("MlpPolicy", env_pendulum, verbose=1, device='cpu')
    timesteps = 100_000  # 총 학습 타임스텝
    rewards_pendulum = []
    intervals = []
    eval_env = gym_new.make('InvertedDoublePendulum-v4')
    for step in range(0, timesteps, 5000):
        model_SAC.learn(total_timesteps=5000, reset_num_timesteps=False, progress_bar=False)
        ep_rews = []
        for _ in range(5):
            obs, _ = eval_env.reset()
            total_rew = 0
            done = False
            while not done:
                action, _ = model_SAC.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_rew += reward
                if terminated or truncated:
                    done = True
            ep_rews.append(total_rew)
        mean_rew = np.mean(ep_rews)
        rewards_pendulum.append(mean_rew)
        intervals.append(step + 5000)
        print(f"InvertedDoublePendulum: {step+5000} timesteps, Mean Reward = {mean_rew:.2f}")
    eval_env.close()
    plt.figure(figsize=(8,5))
    plt.plot(intervals, rewards_pendulum, marker='o')
    plt.xlabel('Training Timesteps')
    plt.ylabel('Mean Episode Reward (avg over 5 episodes)')
    plt.title('SAC on InvertedDoublePendulum-v4')
    save_fig("DoublePendulum_SAC_learning.png")
    plt.show()
    env_pendulum.close()

# ===========================================================================
# 8. Main Execution
# ===========================================================================
def main():
    print("===== Blackjack Experiments =====")
    V_vi_bj, policy_vi_bj, vi_deltas_bj, vi_iters_bj = value_iteration_blackjack(gamma=GAMMA_BLACKJACK)
    V_pi_bj, policy_pi_bj, pi_deltas_bj, pi_iters_bj = policy_iteration_blackjack(gamma=GAMMA_BLACKJACK)
    print(f"Blackjack VI iterations: {len(vi_deltas_bj)}, PI iterations: {pi_iters_bj}")
    
    plt.figure(figsize=(6,4))
    plt.plot(vi_deltas_bj, label="VI ΔV")
    plt.plot(range(1, len(pi_deltas_bj)+1), pi_deltas_bj, label="PI ΔV")
    plt.axvline(len(vi_deltas_bj), color='blue', linestyle='--',
                label=f"VI policy stable @ {len(vi_deltas_bj)}")
    plt.axvline(pi_iters_bj, color='orange', linestyle=':',
                label=f"PI converged @ {pi_iters_bj}")
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Max ΔV")
    plt.title("Blackjack VI vs PI Convergence")
    plt.legend()
    plt.tight_layout()
    save_fig("Blackjack_VI_vs_PI_convergence.png")
    
    dealer_vals = list(range(1,11))
    player_vals = list(range(21,3,-1))
    policy_matrix_noace = np.full((22,11), -1)
    policy_matrix_ace = np.full((22,11), -1)
    for s in policy_pi_bj:
        total, dealer, usable = s
        if usable:
            policy_matrix_ace[total, dealer] = policy_pi_bj[s]
        else:
            policy_matrix_noace[total, dealer] = policy_pi_bj[s]
    pm_noace = policy_matrix_noace[4:22, 1:11]
    pm_ace = policy_matrix_ace[4:22, 1:11]
    fig, axes = plt.subplots(1,2, figsize=(12,5), sharey=True)
    ax1, ax2 = axes
    im1 = ax1.imshow(pm_noace, cmap='coolwarm', origin='lower')
    ax1.set_title('No Usable Ace')
    ax1.set_xlabel('Dealer Showing'); ax1.set_ylabel('Player Sum')
    ax1.set_xticks(np.arange(10)); ax1.set_xticklabels(np.arange(1,11))
    ax1.set_yticks(np.arange(18)); ax1.set_yticklabels(np.arange(4,22))
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_ticks([0,1]); cbar1.set_ticklabels(['Stick (0)','Hit (1)'])
    im2 = ax2.imshow(pm_ace, cmap='coolwarm', origin='lower')
    ax2.set_title('Usable Ace')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_xticks(np.arange(10)); ax2.set_xticklabels(np.arange(1,11))
    ax2.set_yticks(np.arange(18)); ax2.set_yticklabels(np.arange(4,22))
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_ticks([0,1])
    plt.suptitle('Blackjack Optimal Policy (0=Stick, 1=Hit)', fontsize=14)
    plt.tight_layout()
    save_fig("Blackjack_policy_heatmap.png")
    plt.show()
    
    print("Training Blackjack with SARSA and Q-Learning...")
    Q_Qlearn_bj, rewards_Q_bj = train_blackjack_q_sarsa(num_episodes=EPISODES_BLACKJACK, alpha=0.1, method="Q")
    Q_SARSA_bj, rewards_S_bj = train_blackjack_q_sarsa(num_episodes=EPISODES_BLACKJACK, alpha=0.1, method="SARSA")
    avg_rewards_Q_bj = moving_average(rewards_Q_bj, window=MA_WINDOW_BLACKJACK)
    avg_rewards_S_bj = moving_average(rewards_S_bj, window=MA_WINDOW_BLACKJACK)
    episodes_bj = np.arange(len(avg_rewards_Q_bj)) + MA_WINDOW_BLACKJACK
    plt.figure(figsize=(8,5))
    plt.plot(episodes_bj, avg_rewards_Q_bj, label='Q-Learning')
    plt.plot(episodes_bj, avg_rewards_S_bj, label='SARSA')
    plt.axhline(0.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (window={} episodes)'.format(MA_WINDOW_BLACKJACK))
    plt.title('Blackjack: SARSA vs Q-Learning Learning Curve')
    plt.legend()
    save_fig("Blackjack_SARSA_vs_Q.png")
    plt.show()
    
    print("===== CartPole Experiments =====")
    print(f"CartPole VI converged in {vi_cp_iterations} iterations (θ={theta_cp_val}).")
    mid_vel_idx = N1 // 2
    mid_angvel_idx = N3 // 2
    policy_slice = np.zeros((N0, N2), dtype=int)
    for i in range(N0):
        for k in range(N2):
            s_idx = cp_state_to_index[(i, mid_vel_idx, k, mid_angvel_idx)]
            policy_slice[i, k] = policy_cp[s_idx]
    plt.figure(figsize=(6,5))
    plt.imshow(policy_slice, origin='lower', cmap='Paired')
    plt.colorbar(ticks=[0,1]).set_ticklabels(['Left (0)','Right (1)'])
    plt.xlabel('Pole Angle (Discrete Index)')
    plt.ylabel('Cart Position (Discrete Index)')
    plt.title('CartPole Policy Slice (Velocities ~0)')
    plt.tight_layout()
    save_fig("CartPole_policy_slice.png")
    plt.show()
    
    print("Training CartPole with SARSA, Q-Learning, and Double Q-Learning...")
    Q_cartpole_Q, rewards_Q_cp = train_cartpole(agent="Q", num_episodes=EPISODES_CARTPOLE, alpha=0.5)
    Q_cartpole_S, rewards_S_cp = train_cartpole(agent="SARSA", num_episodes=EPISODES_CARTPOLE, alpha=0.5)
    Q_cartpole_D, rewards_D_cp = train_cartpole(agent="DoubleQ", num_episodes=EPISODES_CARTPOLE, alpha=0.5)
    avg_rewards_Q_cp = compute_avg_curve(rewards_Q_cp, window=MA_WINDOW_CARTPOLE)
    avg_rewards_S_cp = compute_avg_curve(rewards_S_cp, window=MA_WINDOW_CARTPOLE)
    avg_rewards_D_cp = compute_avg_curve(rewards_D_cp, window=MA_WINDOW_CARTPOLE)
    plt.figure(figsize=(8,5))
    plt.plot(avg_rewards_Q_cp, label='Q-Learning')
    plt.plot(avg_rewards_S_cp, label='SARSA')
    plt.plot(avg_rewards_D_cp, label='Double Q-Learning')
    plt.axhline(195, color='grey', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (window={} episodes)'.format(MA_WINDOW_CARTPOLE))
    plt.title('CartPole: Q-Learning vs SARSA vs Double Q-Learning')
    plt.legend()
    save_fig("CartPole_Q_SARSA_DoubleQ.png")
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(avg_rewards_Q_cp, label='Q-Learning')
    plt.plot(avg_rewards_S_cp, label='SARSA')
    plt.axhline(195, color='grey', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (window={} episodes)'.format(MA_WINDOW_CARTPOLE))
    plt.title('CartPole: SARSA vs Q-Learning')
    plt.legend()
    save_fig("CartPole_SARSA_vs_Q.png")
    plt.show()
    
    print("===== Extra Credit: Inverted Double Pendulum with SAC =====")
    try:
        env_pendulum = gym_new.make('InvertedDoublePendulum-v4')
    except Exception as e:
        print("Failed to load InvertedDoublePendulum-v4. Ensure MuJoCo is installed.")
        env_pendulum = None
    if env_pendulum is not None:
        from stable_baselines3 import SAC
        model_SAC = SAC("MlpPolicy", env_pendulum, verbose=1, device='cpu')
        timesteps = 100_000
        rewards_pendulum = []
        intervals = []
        eval_env = gym_new.make('InvertedDoublePendulum-v4')
        for step in range(0, timesteps, 5000):
            model_SAC.learn(total_timesteps=5000, reset_num_timesteps=False, progress_bar=False)
            ep_rews = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                total_rew = 0
                done = False
                while not done:
                    action, _ = model_SAC.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    total_rew += reward
                    if terminated or truncated:
                        done = True
                ep_rews.append(total_rew)
            mean_rew = np.mean(ep_rews)
            rewards_pendulum.append(mean_rew)
            intervals.append(step + 5000)
            print(f"InvertedDoublePendulum: {step+5000} timesteps, Mean Reward = {mean_rew:.2f}")
        eval_env.close()
        plt.figure(figsize=(8,5))
        plt.plot(intervals, rewards_pendulum, marker='o')
        plt.xlabel('Training Timesteps')
        plt.ylabel('Mean Episode Reward (avg over 5 episodes)')
        plt.title('SAC on InvertedDoublePendulum-v4')
        save_fig("DoublePendulum_SAC_learning.png")
        plt.show()
        env_pendulum.close()
    
if __name__ == "__main__":
    main()

from env import FootballSkillsEnv
import numpy as np
from collections import defaultdict

transition_call_counter = defaultdict(int)

def terminal_check(s):
    return s[2] == 1


def value_iteration_with_time(envr=FootballSkillsEnv,horizon_length=40):
    env = envr(render_mode="gif", degrade_pitch=True)
    gamma=0.95
    num_states=env.grid_size*env.grid_size*2
    num_actions=env.action_space.n

    # here v dependent on time step as well as state, horizon 0 to H (hence h+1 length)
    v= np.zeros((horizon_length+1, num_states), dtype=float)
    
    # action defined till 0 to H-1, hence length H
    policy= np.full((horizon_length, num_states), -1, dtype=int)

    # going backward to compute v (as v(terminal) always 0)
    for t in range(horizon_length - 1, -1, -1):
        for s in range(num_states):
            state = env.index_to_state(s)
            # Terminal states have zero continuation value and no action
            if terminal_check(state):
                v[t,s] = 0.0
                policy[t,s] = -1
                continue

            sx, sy, _ = state
            best_q_function, best_action = -float("inf"), 0

            for action in range(num_actions):

                q_function = 0.0
                # have to pass timestep=t here
                transitions = env.get_transitions_at_time(state, action, time_step=t)
                transition_call_counter["vi_time_dependent"] += 1
                for prob, s_next in transitions:
                    nx, ny, _ = s_next
                    r = env._get_reward((nx, ny), action, (sx, sy))
                    if terminal_check(s_next):
                        v_next = 0.0 
                    else:
                        v_next=v[t + 1, env.state_to_index(s_next)]
                    q_function += prob * (r + gamma * v_next)

                if q_function > best_q_function:
                    best_q_function=q_function 
                    best_action = action

            v[t, s] = best_q_function
            # instead of establishing policy after, can do it here itself
            policy[t, s] = best_action

    return policy, v

def vanilla_value_iteration(envr=FootballSkillsEnv):
    env=envr(render_mode="gif", degrade_pitch=True)
    gamma = 0.95
    threshold = 1e-6

    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n

    # value function initialization
    v = np.zeros(num_states, dtype=float)

    iterations = 0
    while True:
        iterations += 1
        delta = 0.0
        for s in range(num_states):
            state = env.index_to_state(s)
            # value for terminal states again 0
            if terminal_check(state):
                v[s] = 0.0
                continue

            sx, sy, _ = state
            best_q_function = -float('inf')

            for action in range(num_actions):
                q_function = 0.0
                transition_call_counter["vi"] += 1
                transitions = env.get_transitions_at_time(state, action)
                for p, s_next in transitions:
                    nx, ny, _ = s_next
                    r = env._get_reward((nx, ny), action, (sx, sy))
                    if terminal_check(s_next):
                        v_next = 0.0
                    else:
                        v_next = v[env.state_to_index(s_next)]
                    q_function += p * (r + gamma * v_next)
                if q_function > best_q_function:
                    best_q_function = q_function

            delta = max(delta, abs(best_q_function - v[s]))
            v[s] = best_q_function

        if delta < threshold:
            break

    # policy extraction from the optimal value function
    policy = np.full(num_states, -1, dtype=int)
    for s in range(num_states):
        state = env.index_to_state(s)
        if terminal_check(state):
            policy[s] = -1
            continue

        sx, sy, _ = state
        best_action = 0
        best_q_function = -float('inf')
        for action in range(num_actions):
            q_function = 0.0
            transition_call_counter["vi"] += 1
            transitions = env.get_transitions_at_time(state, action)
            for p, s_next in transitions:
                nx, ny, _ = s_next
                r = env._get_reward((nx, ny), action, (sx, sy))
                if terminal_check(s_next):
                    v_next = 0.0
                else:
                    v_next = v[env.state_to_index(s_next)]
                q_function += p * (r + gamma * v_next)
            if q_function > best_q_function:
                best_q_function, best_action = q_function, action
        policy[s] = best_action

    return policy, v, iterations

def run_episode_time_dependent(policy, seed, envr=FootballSkillsEnv):
    
    env = envr(render_mode="gif", degrade_pitch=True)
    obs, _ = env.reset(seed=seed)
    total_reward, done, truncated = 0.0, False, False
    horizon = policy.shape[0]

    for t in range(horizon):
        state=obs
        if done or truncated or terminal_check(state):
            break
        
        s = env.state_to_index(obs)
        a = policy[t, s]
        obs, reward, done, truncated, _ = env.step(a)
        total_reward += reward
    return total_reward


def run_episode_stationary(policy,seed, envr=FootballSkillsEnv):

    env = envr(render_mode="gif", degrade_pitch=True)
    obs, _ = env.reset(seed=seed)
    total_reward, done, truncated = 0.0, False, False

    while not (done or truncated):
        state=obs
        if terminal_check(state):
            break
        s = env.state_to_index(obs)
        a = policy[s]
        obs, reward, done, truncated, _ = env.step(a)
        total_reward += reward
    return total_reward


def evaluate_time_dependent(policy, num_episodes=20, base_seed=0, envr=FootballSkillsEnv):
    rewards = []
    for i in range(num_episodes):
        rewards.append(run_episode_time_dependent(policy, envr=envr, seed=base_seed + i))
    rewards = np.array(rewards, dtype=float)
    mean = float(rewards.mean())
    std  = float(rewards.std(ddof=1))
    return mean, std


def evaluate_stationary(policy, num_episodes=20, base_seed=0, envr=FootballSkillsEnv):
    rewards = []
    for i in range(num_episodes):
        rewards.append(run_episode_stationary(policy, envr=envr, seed=base_seed + i))
    rewards = np.array(rewards, dtype=float)
    mean = float(rewards.mean())
    std  = float(rewards.std(ddof=1))
    return mean, std

def save_policy_gif(policy, filename, envr=FootballSkillsEnv, horizon=40):
    env = envr(render_mode="gif", degrade_pitch=True)
    if isinstance(policy, np.ndarray) and policy.ndim == 1:
        policy = np.tile(policy, (horizon, 1))  # make it 2D
    env.get_gif(policy, filename=filename) 



if __name__=="__main__":
    optimal_policy_vi_time, value_function_vi_time = value_iteration_with_time()
    print("Optimal Policy (Time-Dependent VI):", optimal_policy_vi_time)
    print("Value Function (Time-Dependent VI):", value_function_vi_time)
    print("Total transition calls in Time-Dependent Value Iteration:", transition_call_counter["vi_time_dependent"])
    optimal_policy_vi, value_function_vi, num_iterations_vi = vanilla_value_iteration(envr=FootballSkillsEnv)
    print("Optimal Policy (Vanilla VI):", optimal_policy_vi)
    print("Value Function (Vanilla VI):", value_function_vi)
    print("Number of iterations (Vanilla VI):", num_iterations_vi)
    mean_vi_time, std_vi_time = evaluate_time_dependent(optimal_policy_vi_time, envr=FootballSkillsEnv, num_episodes=20, base_seed=0)
    mean_vi, std_vi = evaluate_stationary(optimal_policy_vi, envr=FootballSkillsEnv, num_episodes=20, base_seed=0)
    print(f"Time-Dependent VI - Mean Reward: {mean_vi_time}, Standard Deviation: {std_vi_time}")
    print(f"Vanilla VI - Mean Reward: {mean_vi}, Standard Deviation: {std_vi}")
    save_policy_gif(optimal_policy_vi_time, filename="degraded_time_dependent_vi.gif", envr=FootballSkillsEnv)
    save_policy_gif(optimal_policy_vi, filename="degraded_vanilla_vi.gif", envr=FootballSkillsEnv)
    # print("Optimal Policy:", optimal_policy_pi)
    # print("Value Function:", value_function_pi)
    # print("Number of iterations:", num_iterations_pi)
    # print("Optimal Policy:", optimal_policy_vi)
    # print("Value Function:", value_function_vi)
    # print("Number of iterations:", num_iterations_vi)
    # print("Total transition calls in Policy Iteration:", transition_call_counter["pi"])
    # print("Total transition calls in Value Iteration:", transition_call_counter["vi"])
    # if np.array_equal(optimal_policy_pi, optimal_policy_vi):
    #     print("Both algorithms yield the same optimal policy")
    # else:
    #     print("Optimal policies differ between algorithms")

    # # Evaluate both policies for 20 episodes with different seeds
    # pi_mean, pi_std, _ = evaluate_policy(optimal_policy_pi, envr=FootballSkillsEnv, num_episodes=20, starting_seed=0)
    # vi_mean, vi_std, _ = evaluate_policy(optimal_policy_vi, envr=FootballSkillsEnv, num_episodes=20, starting_seed=1000)

    # print("Policy Evaluation over 20 episodes:")
    # print(f"Policy Iteration - Mean Reward: {pi_mean}, Standard Deviation: {pi_std
    # print(f"Value Iteration  -> Mean Reward: {vi_mean}, Standard Deviation: {vi_std}")
    # # Save GIFs of both policies for different seeds
    # save_policy_gif(optimal_policy_pi, filename="policy_iteration.gif", seed=10, envr=FootballSkillsEnv)
    # save_policy_gif(optimal_policy_vi, filename="value_iteration.gif", seed=1010, envr=FootballSkillsEnv)
    
    
from env import FootballSkillsEnv
import numpy as np
from collections import defaultdict

transition_call_counter = defaultdict(int)

def terminal_check(s):
    return s[2] == 1


def policy_iteration(envr=FootballSkillsEnv):
    '''
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (dict): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    
    Algorithm:
    1. Initialize arbitrary policy and value function
    2. Policy Evaluation: Iteratively update value function until convergence
    3. Policy Improvement: Update policy greedily based on current values  
    4. Repeat steps 2-3 until policy converges
    
    Key Environment Methods to Use:
    - env.state_to_index(state_tuple): Converts (x, y, has_shot) tuple to integer index
    - env.index_to_state(index): Converts integer index back to (x, y, has_shot) tuple
    - env.get_transitions_at_time(state, action, time_step=None): Default method for accessing transitions.
    - env._is_terminal(state): Check if state is terminal (has_shot=True)
    - env._get_reward(ball_pos, action, player_pos): Get reward for transition
    - env.reset(seed=None): Reset environment to initial state, returns (observation, info)
    - env.step(action): Execute action, returns (obs, reward, done, truncated, info)
    - env.get_gif(policy, seed=20, filename="output.gif"): Generate GIF visualization 
      of policy execution from given seed
    
    Key Env Variables Notes:
    - env.observation_space.n: Total number of states (use env.grid_size^2 * 2)
    - env.action_space.n: Total number of actions (7 actions: 4 movement + 3 shooting)
    - env.grid_size: Total number of rows in the grid
    '''
    gamma=0.95
    threshold=1e-6
    env = envr(render_mode="gif")
    num_states = env.grid_size*env.grid_size*2
    num_actions = env.action_space.n
    # initializing arbitrary policy (take the action 1 in every state)
    policy = np.full(num_states, 1, dtype=int)
    
    for s in range(num_states):
        if terminal_check(env.index_to_state(s)):
            policy[s] = -1   # no action to take in terminal state

    iterations = 0
    while True:
        
        iterations += 1
        # value function of current policy
        v = policy_evaluation_for_pi(policy,  gamma, threshold)

        # policy improvement
        policy_stable = True
        for s in range(num_states):
            state = env.index_to_state(s)
            # if in terminal state, do nothing
            if terminal_check(state):
                policy[s] = -1
                continue

            old_action=policy[s]

            best_a= None 
            best_q_function=-float('inf')
            sx, sy, _ = state
            for a in range(num_actions):
                q_function = 0.0
                transition_call_counter["pi"] += 1
                transitions_possible = env.get_transitions_at_time(state, a)
    

                for p, s_next in transitions_possible:
                    nx, ny, _ = s_next
                    r = env._get_reward((nx, ny), a, (sx, sy))
                    if terminal_check(s_next):
                        v_next = 0.0 
                    else:
                        v_next=v[env.state_to_index(s_next)]
                    q_function += p * (r + gamma * v_next)

                
                if q_function > best_q_function:
                    best_q_function, best_a = q_function, a

            
            policy[s] = best_a
            if policy[s] != old_action:
                policy_stable = False

        
        if policy_stable:
            break

    return policy, v, iterations


def policy_evaluation_for_pi(policy, gamma, threshold, envr=FootballSkillsEnv):
    env=envr(render_mode="gif")
    num_states = env.grid_size*env.grid_size*2
    # initialize value function dont need to check for terminal states as it is already 0
    v=np.zeros(num_states,dtype=float)

    while True:
        delta=0.0
        for s in range(num_states):
            # state in state form (x,y,has_shot)
            state=env.index_to_state(s)
            if terminal_check(state):
                continue
        
            action=policy[s]
            transition_call_counter["pi"] += 1
            transition_dynamics=env.get_transitions_at_time(state, action)
            v_update=0.0
            sx,sy,_=state
            for probability, next_state in transition_dynamics:
                nx,ny,_=next_state
                reward=env._get_reward((nx,ny),action,(sx,sy))
                if terminal_check(next_state):
                    v_next=0.0
                else:
                    v_next=v[env.state_to_index(next_state)]
                v_update += probability*(reward+gamma*v_next)

            delta=max(delta,abs(v_update-v[s]))
            v[s]=v_update

        if delta<threshold:
            break

    return v

def value_iteration(envr=FootballSkillsEnv):
    env = envr(render_mode="gif")
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


def run_episode_using_policy(policy,seed,envr=FootballSkillsEnv):
    
    env = envr(render_mode="gif")
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    truncated = False
    while not (done or truncated):
        state = obs
        # if in terminal state, break
        if terminal_check(state):
            break
        s_idx = env.state_to_index(state)
        action = policy[s_idx]
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward


def evaluate_policy(policy, starting_seed, envr=FootballSkillsEnv, num_episodes=20):
    rewards_across_episodes = []
    for i in range(num_episodes):
        seed_episode = starting_seed+i
        episode_return = run_episode_using_policy(policy, envr=envr, seed=seed_episode)
        rewards_across_episodes.append(episode_return)
    rewards_across_episodes = np.array(rewards_across_episodes, dtype=float)
    mean = float(rewards_across_episodes.mean())
    std = float(rewards_across_episodes.std(ddof=1))
    return mean, std, rewards_across_episodes

def save_policy_gif(policy, filename, seed, envr=FootballSkillsEnv):
    env = envr(render_mode="gif")
    env.get_gif(policy, filename=filename, seed=seed)


if __name__=="__main__":
    optimal_policy_pi, value_function_pi, num_iterations_pi = policy_iteration()
    optimal_policy_vi, value_function_vi, num_iterations_vi = value_iteration()
    print("Optimal Policy:", optimal_policy_pi)
    print("Value Function:", value_function_pi)
    print("Number of iterations:", num_iterations_pi)
    print("Optimal Policy:", optimal_policy_vi)
    print("Value Function:", value_function_vi)
    print("Number of iterations:", num_iterations_vi)
    print("Total transition calls in Policy Iteration:", transition_call_counter["pi"])
    print("Total transition calls in Value Iteration:", transition_call_counter["vi"])
    if np.array_equal(optimal_policy_pi, optimal_policy_vi):
        print("Both algorithms yield the same optimal policy")
    else:
        print("Optimal policies differ between algorithms")

    # Evaluate both policies for 20 episodes with different seeds
    pi_mean, pi_std, _ = evaluate_policy(optimal_policy_pi, envr=FootballSkillsEnv, num_episodes=20, starting_seed=0)
    vi_mean, vi_std, _ = evaluate_policy(optimal_policy_vi, envr=FootballSkillsEnv, num_episodes=20, starting_seed=1000)

    print("Policy Evaluation over 20 episodes:")
    print(f"Policy Iteration - Mean Reward: {pi_mean}, Standard Deviation: {pi_std}")
    print(f"Value Iteration  -> Mean Reward: {vi_mean}, Standard Deviation: {vi_std}")
    # Save GIFs of both policies for different seeds
    save_policy_gif(optimal_policy_pi, filename="policy_iteration.gif", seed=10, envr=FootballSkillsEnv)
    save_policy_gif(optimal_policy_vi, filename="value_iteration.gif", seed=1010, envr=FootballSkillsEnv)
    
    
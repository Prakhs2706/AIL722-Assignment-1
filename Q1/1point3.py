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
            policy[t, s] = best_action

    return policy, v




if __name__=="__main__":
    optimal_policy_vi, value_function_vi = value_iteration_with_time()
    print("Optimal Policy (Time-Dependent VI):", optimal_policy_vi)
    print("Value Function (Time-Dependent VI):", value_function_vi)
    print("Total transition calls in Time-Dependent Value Iteration:", transition_call_counter["vi_time_dependent"])
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
    # print(f"Policy Iteration - Mean Reward: {pi_mean}, Standard Deviation: {pi_std}")
    # print(f"Value Iteration  -> Mean Reward: {vi_mean}, Standard Deviation: {vi_std}")
    # # Save GIFs of both policies for different seeds
    # save_policy_gif(optimal_policy_pi, filename="policy_iteration.gif", seed=10, envr=FootballSkillsEnv)
    # save_policy_gif(optimal_policy_vi, filename="value_iteration.gif", seed=1010, envr=FootballSkillsEnv)
    
    
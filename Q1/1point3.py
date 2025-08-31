from env import FootballSkillsEnv
import numpy as np
from collections import defaultdict

transition_call_counter = defaultdict(int)

def terminal_check(s):
    return s[2] == 1


def value_iteration_with_time(envr=FootballSkillsEnv,horizon_length=40):
    env = envr(render_mode="gif", degrade_pitch=True)

    num_states=env.grid_size*env.grid_size*2
    num_actions=env.action_space.n

    # V[t, s] is value with t steps-to-go (0..H); terminal at H is zero by construction
    V = np.zeros((horizon_length + 1, num_states), dtype=float)
    # Time-dependent (non-stationary) policy π_t(s)
    policy_t = np.full((horizon_length, num_states), -1, dtype=int)

    # Backward induction: t = H-1, H-2, ..., 0
    for t in range(horizon - 1, -1, -1):
        for s in range(num_states):
            state = env.index_to_state(s)

            # Terminal states have zero continuation value and no action
            if terminal_check(state):
                V[t, s] = 0.0
                policy_t[t, s] = -1
                continue

            sx, sy, _ = state
            best_q, best_a = -float("inf"), 0

            for a in range(num_actions):
                # Count calls if you're tracking
                transition_call_counter["vi_t"] += 1

                q = 0.0
                # Non-stationary dynamics at explicit time step t
                transitions = env.get_transitions_at_time(state, a, time_step=t)
                for p, s_next in transitions:
                    nx, ny, _ = s_next
                    r = env._get_reward((nx, ny), a, (sx, sy))
                    # Next value uses V at (t+1)
                    v_next = 0.0 if terminal_check(s_next) else V[t + 1, env.state_to_index(s_next)]
                    q += p * (r + gamma * v_next)

                if q > best_q:
                    best_q, best_a = q, a

            V[t, s] = best_q
            policy_t[t, s] = best_a

    return policy_t, V, horizon




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

    # # Evaluate both policies for 20 episodes with different seeds
    # pi_mean, pi_std, _ = evaluate_policy(optimal_policy_pi, envr=FootballSkillsEnv, num_episodes=20, starting_seed=0)
    # vi_mean, vi_std, _ = evaluate_policy(optimal_policy_vi, envr=FootballSkillsEnv, num_episodes=20, starting_seed=1000)

    # print("Policy Evaluation over 20 episodes:")
    # print(f"Policy Iteration - Mean Reward: {pi_mean}, Standard Deviation: {pi_std}")
    # print(f"Value Iteration  -> Mean Reward: {vi_mean}, Standard Deviation: {vi_std}")
    # # Save GIFs of both policies for different seeds
    # save_policy_gif(optimal_policy_pi, filename="policy_iteration.gif", seed=10, envr=FootballSkillsEnv)
    # save_policy_gif(optimal_policy_vi, filename="value_iteration.gif", seed=1010, envr=FootballSkillsEnv)
    
    
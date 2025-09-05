import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns


import numpy as np

class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4):
        self.env=env
        self.gamma=gamma
        self.epsilon=epsilon
        self.horizon_length=env.step_limit
        self.max_weight=env.max_weight
        self.probs_array=env.item_probs
        self.values_array=env.item_values
        self.weights_array=env.item_weights
        self.v=None
        
    
    def get_reward_and_done(self, current_weight, item_idx, action):
        item_weight=self.weights_array[item_idx]
        item_value=self.values_array[item_idx]
        if action==1:
            if current_weight+item_weight>self.max_weight:
                reward=0
                done=True
            else:
                reward=item_value
                done=(current_weight+item_weight==self.max_weight)
        else:
            reward=0
            done=False
            
        return reward, done
        

    def value_iteration(self, max_iterations=1000):
        horizon_length=self.horizon_length
        max_weight=self.max_weight
        weights_array=self.weights_array
        values_array=self.values_array
        probs_array=self.probs_array
        v=np.zeros((horizon_length+1, max_weight+1))
        for t in range(horizon_length-1,-1,-1):
            future_v=v[t+1]
            # looping over all possible states (max_weight)
            for weight in range(max_weight+1):
                updated_weight=0.0
                # as 2 actions consider rejecting it too
                reject_val = future_v[weight]
                # the weights that can appear
                for j in range(len(weights_array)):
                    current_weight=weights_array[j]
                    current_value=values_array[j]
                    if weight+current_weight<=max_weight:
                        accept_val=current_value+future_v[weight+current_weight]
                    else:
                        accept_val=0.0
                    updated_weight+=probs_array[j]*max(accept_val, reject_val)
                v[t,weight]=updated_weight
                
        self.v=v
        return v
                

    def get_action(self, state):
        # Support masked dict observation
        if isinstance(state, dict) and 'state' in state:
            state = state['state']
        current_weight = state[0]
        current_item = state[1]
        current_item_weight=state[2]
        current_item_value=state[3]
        t=self.env.step_counter
        if t>=self.horizon_length:
            return 0  # no steps left
        if current_weight + current_item_weight > self.max_weight:
            return 0
        reject=self.v[t + 1, current_weight]
        accept=current_item_value + self.v[t + 1, current_weight + current_item_weight]
        if accept>reject:
            return 1
        else:
            return 0
    



class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4, eval_iterations=1000):
        self.env=env
        self.gamma=gamma
        self.epsilon=epsilon
        self.horizon_length=env.step_limit
        self.max_weight=env.max_weight
        self.probs_array=env.item_probs
        self.values_array=env.item_values
        self.weights_array=env.item_weights
        self.v=None
        self.policy=np.zeros((self.horizon_length,self.max_weight+1,len(self.weights_array)), dtype=float)
        

    def get_reward_and_done(self, current_weight, item_idx, action):
        item_weight=self.weights_array[item_idx]
        item_value=self.values_array[item_idx]
        if action==1:
            if current_weight+item_weight>self.max_weight:
                reward=0
                done=True
            else:
                reward=item_value
                done=(current_weight+item_weight==self.max_weight)
        else:
            reward=0
            done=False
            
        return reward, done
        
    
    def policy_evaluation(self):
        horizon_length=self.horizon_length
        max_weight=self.max_weight
        weights_array=self.weights_array
        values_array=self.values_array
        probs_array=self.probs_array
        v=np.zeros((horizon_length + 1, max_weight + 1))
        for t in range(horizon_length - 1, -1, -1):
            future_v=v[t + 1]
            current_pi=self.policy[t]
            for weight in range(max_weight + 1):
                expected_v=0.0
                row=current_pi[weight]
                for j in range(len(weights_array)):
                    wj=weights_array[j]
                    vj=values_array[j]
                    if row[j]==1:
                        if weight+wj<=max_weight:
                            val=vj+future_v[weight + wj]
                        else:
                            val=0.0
                    else:
                        val=future_v[weight]
                    expected_v+=probs_array[j] * val
                v[t, weight]=expected_v

        self.v = v
        return v
                
        

    def policy_improvement(self):
        horizon_length = self.horizon_length
        max_weight = self.max_weight
        weights_array = self.weights_array
        values_array = self.values_array

        policy_stable = True
        for t in range(horizon_length):
            next_v = self.v[t + 1]
            for w in range(max_weight + 1):
                for j in range(len(weights_array)):
                    wj = weights_array[j]
                    vj = values_array[j]
                    reject_val = next_v[w]
                    if w + wj <= max_weight:
                        accept_val = vj + next_v[w + wj]
                    else:
                        accept_val = 0.0
                    best_action = 1 if accept_val > reject_val else 0
                    if best_action != self.policy[t, w, j]:
                        policy_stable = False
                    self.policy[t, w, j] = best_action
        return policy_stable
                    
            

        

    def run_policy_iteration(self, max_iterations=1000):
        iterations=0
        horizon_length=self.horizon_length
        max_weight=self.max_weight
        weights_array=self.weights_array

        if self.policy is None:
            self.policy=np.zeros((horizon_length, max_weight + 1, len(weights_array)), dtype=float)

        while iterations<max_iterations:
            self.policy_evaluation()
            policy_stable=self.policy_improvement()
            iterations+=1
            if policy_stable:
                break
        return self.v

    def get_action(self, state):
        if isinstance(state, dict) and 'state' in state:
            s = state['state']
        else:
            s = state

        current_weight=s[0]
        current_item=s[1]
        current_item_weight=s[2]

        t = min(self.env.step_counter,self.horizon_length-1)
        if current_weight+current_item_weight>self.max_weight:
            return 0
        return self.policy[t,current_weight,current_item]


def evaluate_policy(env, agent, seed, render_trajectory=False):
    
    np.random.seed(seed)
    env.set_seed(seed)
    
    state=env._RESET()
    total_value=0
    curve=[]
    for _ in range(env.step_limit):
        # Print the weight and value of the currently presented item
        if isinstance(state, dict) and 'state' in state:
            obs = state['state']
        else:
            obs = state
        current_item_weight = obs[2]
        current_item_value = obs[3]
        # print(f"Step {_}: Presented item: weight={current_item_weight}, value={current_item_value}")
        action=agent.get_action(state)
        state,reward,done,_=env.step(action)
        if done:
            break
        total_value+=reward
        curve.append(total_value)
    
    return total_value,curve


def plot_trajectories(seed_to_curve):
    plt.figure(figsize=(9, 5))
    for seed, curve in seed_to_curve.items():
        plt.plot(curve, label=f"Seed {seed}")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Value")
    plt.title("Online Knapsack — Cumulative Value over 50 Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_value_heatmaps(agent):
    T, B = agent.horizon_length, agent.max_weight
    W = agent.weights_array
    Vals = agent.values_array
    v = agent.v

    # final value at t=0 uses continuation v[1]
    baseV = v[1]

    # value of each state (w, j) under the optimal action at t=0
    value_mat = np.zeros((B + 1, len(W)), dtype=float)
    for w in range(B + 1):
        reject_val = baseV[w]
        for j in range(len(W)):
            wj = int(W[j])
            vj = float(Vals[j])
            if w + wj <= B:
                accept_val = vj + baseV[w + wj]
            else:
                accept_val = -np.inf
            value_mat[w, j] = max(reject_val, accept_val)

    # sort x-axis keys in increasing order
    order_by_w = np.argsort(W)
    order_by_v = np.argsort(Vals)
    ratio = W / np.maximum(Vals, 1e-5)
    order_by_ratio = np.argsort(ratio)

    data_w = value_mat[:, order_by_w]
    data_v = value_mat[:, order_by_v]
    data_ratio = value_mat[:, order_by_ratio]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Final Value Function at t=0 (Optimal Action)")

    sns.heatmap(data_w, ax=axes[0], cmap="viridis", cbar_kws={'label': 'State Value'}, xticklabels=False, yticklabels=50)
    axes[0].set_title("X: weight (increasing)")
    axes[0].set_xlabel("item weight")
    axes[0].set_ylabel("current knapsack weight")

    sns.heatmap(data_v, ax=axes[1], cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    axes[1].set_title("X: value (increasing)")
    axes[1].set_xlabel("item value")

    sns.heatmap(data_ratio, ax=axes[2], cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    axes[2].set_title("X: weight/value (increasing)")
    axes[2].set_xlabel("weight/value ratio")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # part 1 for value iteration
    seeds=[0, 1, 2, 3, 4]
    seed_to_curve={}

    for seed in seeds:
        np.random.seed(seed)
        env=OnlineKnapsackEnv()
        agent=ValueIterationOnlineKnapsack(env)
        agent.value_iteration()
        total_value, curve=evaluate_policy(env, agent, seed)
        seed_to_curve[seed]=curve
        print(f"Seed {seed}, total knapsack value={total_value}")

    # trajectories
    plot_trajectories(seed_to_curve)

    # heatmaps
    plot_value_heatmaps(agent)
    
    
    # part 2 for policy iteration
    env=OnlineKnapsackEnv()
    agent=PolicyIterationOnlineKnapsack(env)
    agent.run_policy_iteration()
        # Policy iteraton evals for different seeds
    seeds = [0, 1, 2, 3, 4]
    pi_seed_to_curve = {}

    for seed in seeds:
        np.random.seed(seed)
        env = OnlineKnapsackEnv()
        pi_agent = PolicyIterationOnlineKnapsack(env)
        pi_agent.run_policy_iteration()
        total_value, curve = evaluate_policy(env, pi_agent, seed)
        pi_seed_to_curve[seed] = curve
        print(f"[PI] Seed {seed}: total knapsack value = {total_value}")

    # Trajectories for PI
    plot_trajectories(pi_seed_to_curve)

    # Heatmaps for PI (use the last trained agent)
    plot_value_heatmaps(pi_agent)
    
    # for part3, just change the env.step_limit to 5.50,100 in knapsack.py for part 3
    np.random.seed(0)
    env = OnlineKnapsackEnv()
    agent = ValueIterationOnlineKnapsack(env)
    agent.value_iteration()
    total_value, curve = evaluate_policy(env, agent, seed=0)
    print(f"total knapsack value={total_value}")
    plot_value_heatmaps(agent)
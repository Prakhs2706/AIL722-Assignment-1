import numpy as np
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import mode
import sys
import copy
import time
import random

random.seed(42)       # Set seed for Python's built-in random module
np.random.seed(42) 
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

actions = np.array([-2, -1, 0, 1, 2], dtype=int)
def _max_cash_possible(initial_cash, holding_limit, prices, horizon_length, sell_cost=1):
    max_price = np.max(prices)
    return int(initial_cash + holding_limit * max(0, max_price - sell_cost))

def value_iteration(prices,initial_cash,holding_limit,buy_cost,sell_cost,lot_size,gamma,max_cash_possible):
    prices=np.asarray(prices, dtype=int)
    horizon_length=len(prices)
    if max_cash_possible is None:
        max_cash_possible=_max_cash_possible(initial_cash,holding_limit, prices, horizon_length)
    # state space is basically timestep, cash, holdings
    v=np.zeros((horizon_length, max_cash_possible+1, holding_limit+1),dtype=float)
    policy=np.zeros((horizon_length, max_cash_possible+1, holding_limit+1),dtype=int)

    for t in range(horizon_length - 1, -1, -1):
        p=prices[t]
        buy_unit_cost=p+buy_cost
        sell_unit_gain=p-sell_cost
        for cash in range(max_cash_possible+1):
            for holding in range(holding_limit + 1):
                best_val=-float('inf')
                # placeholder for best action
                best_action=0
                for action in actions:
                    if action==0:
                        a_eff=0
                        h2=holding
                        c2=cash
                    elif action>0:
                        # action cant be more than the lot size
                        a_req=min(action,lot_size)
                        # can't exceed the holding limit
                        a_req=min(a_req, holding_limit - holding)
                        if a_req<= 0:
                            a_eff=0
                            h2=holding
                            c2=cash
                        else:
                            # we can buy now
                            if buy_unit_cost<=0: # sanity check
                                max_aff = 0
                            else:
                                max_aff=cash//buy_unit_cost
                            a_eff=min(a_req,max_aff)
                            if a_eff<=0:
                                h2=holding
                                c2=cash
                            else:
                                c2=cash-a_eff*buy_unit_cost
                                h2=holding+a_eff
                    else:  # selling here
                        a_req=min(-action,lot_size)
                        sell_amt=min(a_req,holding)
                        a_eff=-sell_amt
                        if sell_amt<=0:
                            h2=holding
                            c2=cash
                        else:
                            c2=cash+sell_amt*sell_unit_gain
                            h2=holding-sell_amt

                    c2=max(0,min(c2, max_cash_possible))
                    h2=max(0, min(h2, holding_limit))

                    # Evaluate
                    if t==horizon_length - 1:
                        # Terminal check
                        cand = c2+p*(h2)
                    else:
                        # Non-terminal uses value function
                        cand=gamma*v[t+1,c2,h2]

                    if cand>best_val:
                        best_val=cand
                        best_action=action if a_eff!= 0 or action == 0 else 0

                v[t,cash,holding]=best_val
                policy[t,cash,holding]=best_action


    return v, policy, max_cash_possible




def simulate_episode(env, policy, max_cash_possible):
    state = env.reset()
    prices = env.asset_prices[0]
    T = env.step_limit

    wealth_hist, cash_hist, hold_hist, price_hist, action_hist = [], [], [], [], []
    total_reward = 0.0

    for t in range(T):
        price_t = int(prices[t])
        price_hist.append(price_t)

        # Read current cash/holding from env to index policy
        cash_t = int(env.cash)
        try:
            holding_t = int(env.holdings[0])
        except Exception:
            holding_t = int(env.holdings)

        c_idx = max(0, min(cash_t, max_cash_possible))
        h_idx = max(0, min(holding_t, env.holding_limit[0] if hasattr(env, 'holding_limit') else holding_t))

        a = int(policy[t, c_idx, h_idx])
        action_hist.append(a)

        next_state, r, done, _ = env.step(np.array([a], dtype=np.int32))
        total_reward += float(r)

        cash_after = float(env.cash)
        try:
            hold_after = int(env.holdings[0])
        except Exception:
            hold_after = int(env.holdings)

        wealth_after = float(cash_after + price_t * hold_after)
        cash_hist.append(cash_after)
        hold_hist.append(hold_after)
        wealth_hist.append(wealth_after)

        if done:
            break

    return {
        'prices': np.array(price_hist, dtype=float),
        'cash': np.array(cash_hist, dtype=float),
        'holdings': np.array(hold_hist, dtype=float),
        'wealth': np.array(wealth_hist, dtype=float),
        'actions': np.array(action_hist, dtype=int),
        'final_reward': total_reward,
    }


def plot_trajectory(ts, title, outfile):
    t = np.arange(1, len(ts['wealth']) + 1)
    plt.figure(figsize=(8, 4.6))
    plt.plot(t, ts['wealth'], label='Wealth')
    plt.plot(t, ts['cash'], label='Cash')
    plt.plot(t, ts['holdings'], label='Holdings')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()


def plot_progress(values, title, outfile):
    x = np.arange(1, len(values) + 1)
    plt.figure(figsize=(7.6, 4.4))
    plt.plot(x, values, marker='o')
    plt.xlabel('Episode index')
    plt.ylabel('End wealth')
    plt.title(title)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def plot_convergence(diffs, title, outfile):
    x = np.arange(1, len(diffs) + 1)
    plt.figure(figsize=(7.6, 4.4))
    plt.plot(x, diffs, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Max Value Difference')
    plt.title(title)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

# policy iteration starts from here

def _compute_transition(cash, holding, action, price, buy_cost, sell_cost, lot_size, holding_limit, max_cash_possible):
    p=price
    buy_unit_cost=p+buy_cost
    sell_unit_gain=p-sell_cost

    if action==0:
        a_eff=0
        h2=holding
        c2=cash
    elif action > 0:
        a_req=min(action, lot_size)
        a_req=min(a_req, holding_limit - holding)
        if a_req<=0:
            a_eff=0
            h2=holding
            c2=cash
        else:
            if buy_unit_cost<=0:
                max_aff=0
            else:
                max_aff=cash//buy_unit_cost
            a_eff=min(a_req, max_aff)
            if a_eff<=0:
                h2=holding
                c2=cash
            else:
                c2=cash-a_eff*buy_unit_cost
                h2=holding+a_eff
    else:  # selling time now
        a_req=min(-action, lot_size)
        sell_amt=min(a_req, holding)
        a_eff=-sell_amt
        if sell_amt<=0:
            h2=holding
            c2=cash
        else:
            c2=cash+sell_amt*sell_unit_gain
            if c2>max_cash_possible:
                c2=max_cash_possible
            h2=holding-sell_amt

    c2=max(0, min(c2, max_cash_possible))
    h2=max(0, min(h2, holding_limit))
    return c2, h2, a_eff


def policy_iteration(prices, initial_cash, holding_limit, buy_cost, sell_cost, lot_size, gamma, max_cash_possible=None):
    prices = np.asarray(prices, dtype=int)
    horizon_length = len(prices)

    if max_cash_possible is None:
        max_cash_possible=_max_cash_possible(initial_cash, holding_limit, prices, horizon_length)

    v=np.zeros((horizon_length, max_cash_possible + 1, holding_limit + 1), dtype=float)
    policy=np.zeros((horizon_length, max_cash_possible + 1, holding_limit + 1), dtype=int)
    policy_stable=False
    tolerance=1e-6
    delta=float('inf')
    
    while not policy_stable:
        # Policy Evaluation
        delta = float('inf')
        while delta > tolerance:
            delta=0.0
            for t in range(horizon_length-1,-1,-1):
                p=prices[t]
                for cash in range(max_cash_possible+1):
                    for holding in range(holding_limit+1):
                        a=policy[t,cash,holding]
                        c2,h2,_=_compute_transition(cash, holding, a, p, buy_cost, sell_cost, lot_size, holding_limit, max_cash_possible)
                        if t==horizon_length-1:
                            # terminal case handling
                            new_v=c2+p*h2
                        else:
                            new_v=gamma*v[t+1,c2,h2]
                        delta=max(delta, abs(new_v - v[t, cash, holding]))
                        v[t,cash,holding]=new_v
            
        # Policy Improvement
        policy_stable=True
        for t in range(horizon_length-1,-1,-1):
            p=prices[t]
            for cash in range(max_cash_possible+1):
                for holding in range(holding_limit + 1):
                    old_a=policy[t,cash,holding]
                    best_val=-float('inf')
                    best_a=old_a
                    for a in actions:
                        c2,h2,a_eff=_compute_transition(cash, holding, a, p, buy_cost, sell_cost, lot_size, holding_limit, max_cash_possible)
                        if t==horizon_length - 1:
                            cand=c2+p*h2
                        else:
                            cand=gamma*v[t + 1, c2, h2]
                        if cand>best_val:
                            best_val=cand
                            best_a= a if a_eff != 0 or a == 0 else 0
                    policy[t, cash, holding]=best_a
                    if best_a!=old_a:
                        policy_stable = False

    return v, policy, max_cash_possible

def policy_iteration_with_tracking(prices, initial_cash, holding_limit, buy_cost, sell_cost, lot_size, gamma, tol=1e-2, max_iter=1000, max_cash_possible=None):
    prices = np.asarray(prices, dtype=int)
    horizon_length = len(prices)

    if max_cash_possible is None:
        max_cash_possible=_max_cash_possible(initial_cash, holding_limit, prices, horizon_length)

    v=np.zeros((horizon_length, max_cash_possible + 1, holding_limit + 1), dtype=float)
    policy=np.zeros((horizon_length, max_cash_possible + 1, holding_limit + 1), dtype=int)
    diffs = [] # This will store the max change in V over a *policy iteration*
    
    for iteration in range(max_iter):
        # Policy Evaluation
        v_prev = v.copy()   # snapshot for diff tracking
        policy_eval_delta = float('inf')
        while policy_eval_delta > tol: # Use tol for policy evaluation
            policy_eval_delta = 0.0
            for t in range(horizon_length-1,-1,-1):
                p=prices[t]
                for cash in range(max_cash_possible+1):
                    for holding in range(holding_limit+1):
                        a=policy[t,cash,holding]
                        c2,h2,_=_compute_transition(cash, holding, a, p, buy_cost, sell_cost, lot_size, holding_limit, max_cash_possible)
                        
                        old_v = v[t, cash, holding] # Store current V for delta calculation
                        if t==horizon_length-1:
                            new_v=c2+p*h2
                        else:
                            new_v=gamma*v[t+1,c2,h2]
                        
                        policy_eval_delta=max(policy_eval_delta, abs(new_v - old_v)) # Calculate delta
                        v[t,cash,holding]=new_v

        # Record the actual change in value function at this policy-iteration step
        diff_iter = np.max(np.abs(v - v_prev))
        diffs.append(diff_iter)

        # Policy Improvement
        policy_stable=True
        for t in range(horizon_length-1,-1,-1):
            p=prices[t]
            for cash in range(max_cash_possible+1):
                for holding in range(holding_limit + 1):
                    old_a=policy[t,cash,holding]
                    best_val=-float('inf')
                    best_a=old_a
                    for a in actions:
                        c2,h2,a_eff=_compute_transition(cash, holding, a, p, buy_cost, sell_cost, lot_size, holding_limit, max_cash_possible)
                        if t==horizon_length - 1:
                            cand=c2+p*h2
                        else:
                            cand=gamma*v[t + 1, c2, h2]
                        if cand>best_val:
                            best_val=cand
                            best_a= a if a_eff != 0 or a == 0 else 0
                    policy[t, cash, holding]=best_a
                    if best_a!=old_a:
                        policy_stable = False

        if policy_stable:
            # If the policy is stable, we have converged.
            # No need to run policy evaluation again, break.
            break

    return v, policy, diffs


if __name__=="__main__":
    start_time=time.time()


    ###Part 1 and Part 2
    ####Please train the value and policy iteration training algo for the given three sequences of prices
    ####Config1
    env = DiscretePortfolioOptEnv(prices=[1, 3, 5, 5 , 4, 3, 2, 3, 5, 8])

    ####Config2
    env = DiscretePortfolioOptEnv(prices=[2, 2, 2, 4 ,2, 2, 4, 2, 2, 2])

    ####Config3
    env = DiscretePortfolioOptEnv(prices=[4, 1, 4, 1 ,4, 4, 4, 1, 1, 4])


    # Run Value Iteration on the three deterministic price sequences for both gammas
    price_sets = [
        ("Config1", [1, 3, 5, 5, 4, 3, 2, 3, 5, 8]),
        ("Config2", [2, 2, 2, 4, 2, 2, 4, 2, 2, 2]),
        ("Config3", [4, 1, 4, 1, 4, 4, 4, 1, 1, 4]),
    ]

    for gamma in (0.999, 1.0):
        end_wealths = []
        for cfg_name, price_seq in price_sets:
            env_det = DiscretePortfolioOptEnv(prices=price_seq)
            initial_cash = int(env_det.initial_cash)
            try:
                holding_limit = int(env_det.holding_limit[0])
            except Exception:
                holding_limit = int(env_det.holding_limit)
            try:
                buy_cost = int(env_det.buy_cost[0])
                sell_cost = int(env_det.sell_cost[0])
            except Exception:
                buy_cost = int(env_det.buy_cost)
                sell_cost = int(env_det.sell_cost)
            lot_size = int(env_det.lot_size)

            max_cash_possible = None
            v, policy, max_cash_possible = value_iteration(
                prices=price_seq,
                initial_cash=initial_cash,
                holding_limit=holding_limit,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                lot_size=lot_size,
                gamma=gamma,
                max_cash_possible=max_cash_possible,
            )

            traj = simulate_episode(env_det, policy, max_cash_possible)
            end_wealths.append(float(traj['final_reward']))

            # Save trajectory figure for this config/gamma
            out_file = f"vi_{cfg_name}_gamma{gamma}_trajectory.png"
            plot_trajectory(traj, title=f"{cfg_name} – VI Trajectory (gamma={gamma})", outfile=out_file)
            print(f"Saved trajectory: {out_file} | Final wealth = {traj['final_reward']:.2f}")

        # Progress over the configs for this gamma
        prog_file = f"vi_training_progress_gamma{gamma}.png"
        plot_progress(end_wealths, title=f"Value Iteration End-Wealth (gamma={gamma})", outfile=prog_file)
        
        print(f"Saved training progress: {prog_file}")

    # Policy Iteration
    pi_start = time.time()
    for gamma in (0.999, 1.0):
        end_wealths_pi = []
        for cfg_name, price_seq in price_sets:
            env_det = DiscretePortfolioOptEnv(prices=price_seq)
            initial_cash = int(env_det.initial_cash)
            try:
                holding_limit = int(env_det.holding_limit[0])
            except Exception:
                holding_limit = int(env_det.holding_limit)
            try:
                buy_cost = int(env_det.buy_cost[0])
                sell_cost = int(env_det.sell_cost[0])
            except Exception:
                buy_cost = int(env_det.buy_cost)
                sell_cost = int(env_det.sell_cost)
            lot_size = int(env_det.lot_size)

            v_pi, policy_pi, max_cash_possible_pi = policy_iteration(
                prices=price_seq,
                initial_cash=initial_cash,
                holding_limit=holding_limit,
                buy_cost=buy_cost,
                sell_cost=sell_cost,
                lot_size=lot_size,
                gamma=gamma,
                max_cash_possible=None,
            )

            traj_pi = simulate_episode(env_det, policy_pi, max_cash_possible_pi)
            end_wealths_pi.append(float(traj_pi['final_reward']))

            out_file_pi = f"pi_{cfg_name}_gamma{gamma}_trajectory.png"
            plot_trajectory(traj_pi, title=f"{cfg_name} – PI Trajectory (gamma={gamma})", outfile=out_file_pi)
            print(f"[PI] Saved trajectory: {out_file_pi} | Final wealth = {traj_pi['final_reward']:.2f}")

        prog_file_pi = f"pi_training_progress_gamma{gamma}.png"
        plot_progress(end_wealths_pi, title=f"Policy Iteration End-Wealth (gamma={gamma})", outfile=prog_file_pi)
        print(f"[PI] Saved training progress: {prog_file_pi}")

    pi_elapsed = time.time() - pi_start
    print(f"[PI] Execution time: {pi_elapsed:.3f} seconds")

    # Variance 1.0 experiment WITHOUT explicit prices
    env_var = DiscretePortfolioOptEnv(variance=1.0)

    initial_cash = int(env_var.initial_cash)
    try:
        holding_limit = int(env_var.holding_limit[0])
    except Exception:
        holding_limit = int(env_var.holding_limit)
    try:
        buy_cost = int(env_var.buy_cost[0])
        sell_cost = int(env_var.sell_cost[0])
    except Exception:
        buy_cost = int(env_var.buy_cost)
        sell_cost = int(env_var.sell_cost)
    lot_size = int(env_var.lot_size)

    # Use the environment's auto-generated price trajectory
    prices_var = env_var.asset_prices[0]
    print("Generated stochastic prices:", prices_var)

    v_var, policy_var, diffs_var = policy_iteration_with_tracking(
        prices=prices_var,
        initial_cash=initial_cash,
        holding_limit=holding_limit,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        lot_size=lot_size,
        gamma=1.0,
        tol=1e-2,
        max_iter=1000
    )

    plot_convergence(
        diffs_var,
        title="Policy Iteration Convergence (Variance=1.0, gamma=1.0)",
        outfile="pi_convergence_variance1.png"
    )
    print("Saved convergence plot for variance=1.0 as pi_convergence_variance1.png")
    
    best_v0 = v_var[0, min(len(v_var[0]) - 1, initial_cash), 0]   # value of initial state
    pi_end_wealth = simulate_episode(env_var, policy_var, max_cash_possible=max_cash_possible_pi)['final_reward']

    print("\n==========  VARIANCE = 1.0  (single stochastic trajectory)  ==========")
    print(f"Initial cash                       : {initial_cash}")
    print(f"Price path                         : {prices_var.tolist()}")
    print(f"Best V[0, cash=20, holding=0]      : {best_v0:.2f}")
    print(f"End wealth following PI policy     : {pi_end_wealth:.2f}")
    print(f"Iterations until policy stable     : {len(diffs_var)}")
    print(f"Max-value-diff per iteration (≤5)  : {[round(x,4) for x in diffs_var[:5]]} ...")

    ###Part 3. (Portfolio Optimizaton)
# env = DiscretePortfolioOptEnv(variance=1)

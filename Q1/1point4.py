from env import FootballSkillsEnv
import numpy as np
import heapq
from collections import defaultdict

# -----------------------------
# Utilities and global counters
# -----------------------------

def terminal_check(s):
    # s is a tuple (x, y, has_shot)
    return s[2] == 1

transition_call_counter = defaultdict(int)

# ------------------------------------------------------
# Plain-dict transition storage and predecessor mapping
# ------------------------------------------------------
# transitions[(s_idx, a)] -> list of tuples (p, ns_idx, r, term)
# predecessors[ns_idx] -> set of s_idx that can reach ns_idx under some action

def _state_index(env, s):
    return env.state_to_index(s)


def get_transitions(env, s_idx, a, transitions, predecessors, gamma, counter_key):
    """Fetch transitions for (s_idx, a) into a plain dict once. Counts env calls only once per key.
    Returns list[(p, ns_idx, r, term)].
    """
    key = (s_idx, a)
    if key in transitions:
        return transitions[key]

    s = env.index_to_state(s_idx)
    if counter_key is not None:
        transition_call_counter[counter_key] += 1
    raw = env.get_transitions_at_time(s, a)
    sx, sy, _ = s
    cooked = []
    for p, s_next in raw:
        ns_idx = _state_index(env, s_next)
        nx, ny, _ = s_next
        r = env._get_reward((nx, ny), a, (sx, sy))
        term = terminal_check(s_next)
        cooked.append((p, ns_idx, r, term))
        predecessors[ns_idx].add(s_idx)
    transitions[key] = cooked
    return cooked


def bellman_backup_max_q(env, s_idx, num_actions, V, transitions, predecessors, gamma, counter_key):
    best_q = -float('inf')
    for a in range(num_actions):
        q = 0.0
        for p, ns_idx, r, term in get_transitions(env, s_idx, a, transitions, predecessors, gamma, counter_key):
            v_next = 0.0 if term else V[ns_idx]
            q += p * (r + gamma * v_next)
        if q > best_q:
            best_q = q
    return best_q

# -----------------------------------------
# Baseline Value Iteration (for comparison)
# -----------------------------------------

def value_iteration_basic(envr=FootballSkillsEnv, gamma=0.95, threshold=1e-6):
    env = envr(render_mode="gif")
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n

    V = np.zeros(num_states, dtype=float)

    # Plain dicts
    transitions = {}
    predecessors = defaultdict(set)
    counter_key = 'vi_basic'

    iters = 0
    while True:
        iters += 1
        delta = 0.0
        for s_idx in range(num_states):
            s = env.index_to_state(s_idx)
            if terminal_check(s):
                if V[s_idx] != 0.0:
                    delta = max(delta, abs(V[s_idx]))
                    V[s_idx] = 0.0
                continue
            best_q = bellman_backup_max_q(env, s_idx, num_actions, V, transitions, predecessors, gamma, counter_key)
            delta = max(delta, abs(best_q - V[s_idx]))
            V[s_idx] = best_q
        if delta < threshold:
            break

    # Greedy policy extraction
    policy = np.full(num_states, -1, dtype=int)
    for s_idx in range(num_states):
        s = env.index_to_state(s_idx)
        if terminal_check(s):
            continue
        best_a, best_q = 0, -float('inf')
        for a in range(num_actions):
            q = 0.0
            for p, ns_idx, r, term in get_transitions(env, s_idx, a, transitions, predecessors, gamma, None):
                v_next = 0.0 if term else V[ns_idx]
                q += p * (r + gamma * v_next)
            if q > best_q:
                best_q, best_a = q, a
        policy[s_idx] = best_a

    return policy, V, iters

# -----------------------------------------------------
# Modified VI: Prioritized (Residual-Driven) with dicts
# -----------------------------------------------------

def value_iteration_prioritized(envr=FootballSkillsEnv, gamma=0.95, threshold=1e-6):
    env = envr(render_mode="gif")
    num_states = env.grid_size * env.grid_size * 2
    num_actions = env.action_space.n

    V = np.zeros(num_states, dtype=float)

    # Plain dicts only
    transitions = {}
    predecessors = defaultdict(set)
    counter_key = 'vi_mod'

    def residual(s_idx):
        s = env.index_to_state(s_idx)
        if terminal_check(s):
            return 0.0
        best_q = bellman_backup_max_q(env, s_idx, num_actions, V, transitions, predecessors, gamma, counter_key)
        return abs(best_q - V[s_idx])

    # Seed with only the start state (avoid touching all states up front)
    heap = []  # (-residual, s_idx)
    in_heap = np.zeros(num_states, dtype=bool)
    start_obs, _ = env.reset(seed=0)
    start_idx = env.state_to_index(start_obs)
    if not terminal_check(start_obs):
        res0 = residual(start_idx)
        if res0 > threshold:
            heapq.heappush(heap, (-res0, start_idx))
            in_heap[start_idx] = True

    updates = 0
    while heap:
        neg_res, s_idx = heapq.heappop(heap)
        in_heap[s_idx] = False
        cur_res = -neg_res
        # Recompute to avoid staleness
        new_res = residual(s_idx)
        if new_res + 1e-12 < cur_res and new_res > threshold:
            heapq.heappush(heap, (-new_res, s_idx))
            in_heap[s_idx] = True
            continue
        if new_res <= threshold:
            continue

        # Backup
        s = env.index_to_state(s_idx)
        if not terminal_check(s):
            V[s_idx] = bellman_backup_max_q(env, s_idx, num_actions, V, transitions, predecessors, gamma, counter_key)
            updates += 1

        # Also consider successors of s_idx to grow the explored set forward
        for a in range(num_actions):
            for p, ns_idx, r, term in get_transitions(env, s_idx, a, transitions, predecessors, gamma, counter_key):
                if terminal_check(env.index_to_state(ns_idx)):
                    continue
                res_succ = residual(ns_idx)
                if res_succ > threshold and not in_heap[ns_idx]:
                    heapq.heappush(heap, (-res_succ, ns_idx))
                    in_heap[ns_idx] = True

        # Recompute residuals for predecessors only (already tracked in dict)
        for pred in predecessors.get(s_idx, ()):  # states that can reach s_idx
            if terminal_check(env.index_to_state(pred)):
                continue
            res = residual(pred)
            if res > threshold and not in_heap[pred]:
                heapq.heappush(heap, (-res, pred))
                in_heap[pred] = True

    # Greedy policy
    policy = np.full(num_states, -1, dtype=int)
    for s_idx in range(num_states):
        s = env.index_to_state(s_idx)
        if terminal_check(s):
            continue
        best_a, best_q = 0, -float('inf')
        for a in range(num_actions):
            q = 0.0
            for p, ns_idx, r, term in get_transitions(env, s_idx, a, transitions, predecessors, gamma, None):
                v_next = 0.0 if term else V[ns_idx]
                q += p * (r + gamma * v_next)
            if q > best_q:
                best_q, best_a = q, a
        policy[s_idx] = best_a

    return policy, V, updates

# -----------------
# Command-line run
# -----------------
if __name__ == "__main__":
    # Baseline VI
    pi_basic, V_basic, it_basic = value_iteration_basic()
    calls_basic = transition_call_counter['vi_basic']

    # Prioritized VI
    pi_mod, V_mod, it_mod = value_iteration_prioritized()
    calls_mod = transition_call_counter['vi_mod']

    print("=== Baseline Value Iteration ===")
    print(f"Iterations (sweeps): {it_basic}")
    print(f"Total env.get_transitions_at_time calls: {calls_basic}")

    print("\n=== Prioritized Value Iteration ===")
    print(f"State updates performed: {it_mod}")
    print(f"Total env.get_transitions_at_time calls: {calls_mod}")

    same_policy = np.array_equal(pi_basic, pi_mod)
    print("\nPolicies equal? ", same_policy)
    if not same_policy:
        diff = np.where(pi_basic != pi_mod)[0][:10]
        print("First differing state indices:", diff)

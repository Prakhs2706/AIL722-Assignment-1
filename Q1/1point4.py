from env import FootballSkillsEnv
import numpy as np
import heapq
from collections import defaultdict

transition_call_counter = defaultdict(int)

def terminal_check(s):
        return s[2]==1
    
def get_transitions(env, s_idx, a, trans_dict, pred_dict, counter_key):
        # avoids repeated calls to env
        # as env is stationary, we can cache transitions   
        key=(s_idx, a)
        if key in trans_dict:
            return trans_dict[key]
        s=env.index_to_state(s_idx)
        transition_call_counter[counter_key]+=1
        raw=env.get_transitions_at_time(s, a)
        sx,sy, _=s
        cooked=[]
        for p, s_next in raw:
            ns_idx = env.state_to_index(s_next)
            nx, ny, _ = s_next
            r = env._get_reward((nx, ny), a, (sx, sy))
            term=terminal_check(s_next)
            cooked.append((p, ns_idx, r, term))
            pred_dict[ns_idx].add(s_idx)
        trans_dict[key] = cooked
        return cooked
    
def bellman_max_q(env,s_idx,num_actions,v, trans_dict,pred_dict,gamma):
        best_q = -float('inf')
        for a in range(num_actions):
            q=0.0
            for p, ns_idx, r, term in get_transitions(env, s_idx, a, trans_dict, pred_dict, 'vi_updated'):
                v_next=0.0 if term else v[ns_idx]
                q+=p*(r+gamma*v_next)
            if q>best_q:
                best_q=q
        return best_q

def value_iteration(envr=FootballSkillsEnv):
    # heap is defined over negative residuals to make it a max-heap
    heap=[]  # (-residual, state_index)
    gamma=0.95
    threshold=1e-6
    env=envr(render_mode="gif")
    num_states=env.grid_size*env.grid_size*2
    num_actions=env.action_space.n
    v=np.zeros(num_states, dtype=float)
    transitions={}
    predecessors=defaultdict(set)

    for s_idx in range(num_states):
        s=env.index_to_state(s_idx)
        if terminal_check(s):
            continue
        res=abs(bellman_max_q(env, s_idx, num_actions, v, transitions, predecessors, gamma)-v[s_idx])
        if res>threshold:
            heapq.heappush(heap, (-res, s_idx))
    updates=0
    in_heap=np.zeros(num_states, dtype=bool)
    for _, s_idx in heap:
        in_heap[s_idx]=True

    while heap:
        _, s_idx = heapq.heappop(heap)
        in_heap[s_idx]=False
        # if residual has dropped below threshold, skip
        current_res=abs(bellman_max_q(env, s_idx, num_actions, v, transitions, predecessors, gamma)-v[s_idx])
        if current_res<threshold:
            continue
        # else backup
        v[s_idx]=bellman_max_q(env, s_idx, num_actions, v, transitions, predecessors, gamma)
        updates+=1
        # push predecessors into heap
        for pred in predecessors[s_idx]:
            if terminal_check(env.index_to_state(pred)):
                continue
            res=abs(bellman_max_q(env, pred, num_actions, v, transitions, predecessors, gamma) - v[pred])
            if res>threshold and not in_heap[pred]:
                heapq.heappush(heap, (-res, pred))
                in_heap[pred] = True

    # greedy policy extraction using cached transitions (does not call env again) and saves calls
    policy = np.full(num_states, -1, dtype=int)
    for s_idx in range(num_states):
        s=env.index_to_state(s_idx)
        if terminal_check(s):
            continue
        best_a, best_q = 0, -float('inf')
        for a in range(num_actions):
            q=0.0
            for p, ns_idx, r, term in transitions[(s_idx, a)]:
                v_next=0.0 if term else v[ns_idx]
                q+=p*(r+gamma*v_next)
            if q>best_q:
                best_q, best_a=q, a
        policy[s_idx]=best_a

    return policy, v, updates, transition_call_counter['vi_updated']

        
        
def vanilla_value_iteration(envr=FootballSkillsEnv):
    env=envr(render_mode="gif")
    gamma=0.95
    threshold=1e-6
    num_states=env.grid_size*env.grid_size*2
    num_actions=env.action_space.n
    # value function initialization
    v=np.zeros(num_states, dtype=float)
    policy=np.full(num_states, -1, dtype=int)  # will be filled on the fly

    iterations=0
    while True:
        iterations+=1
        delta=0.0
        for s in range(num_states):
            state=env.index_to_state(s)
            # value for terminal states again 0
            if terminal_check(state):
                v[s] = 0.0
                continue
            sx, sy, _ = state
            best_q_function=-float('inf')
            best_action = -1

            for action in range(num_actions):
                q_function=0.0
                transition_call_counter["vi"]+=1
                transitions=env.get_transitions_at_time(state, action)
                for p, s_next in transitions:
                    nx, ny, _ =s_next
                    r=env._get_reward((nx, ny), action, (sx, sy))
                    if terminal_check(s_next):
                        v_next = 0.0
                    else:
                        v_next=v[env.state_to_index(s_next)]
                    q_function+=p*(r+gamma*v_next)
                if q_function>best_q_function:
                    best_q_function=q_function
                    best_action=action

            delta=max(delta,abs(best_q_function-v[s]))
            v[s]=best_q_function
            # update greedy policy on the fly (same sweep)
            policy[s]=best_action if not terminal_check(state) else -1

        if delta<threshold:
            break
    
    return policy, v, iterations

if __name__=="__main__":
    policy, v, updates, trans_calls = value_iteration()
    print("Number of updates:", updates)
    print("Number of trans calls", trans_calls)
    policy_vanilla, v_vanilla, iters = vanilla_value_iteration()
    print("Vanilla iters:", iters)
    print("number of vanilla trans calls:", transition_call_counter['vi'])
    if np.all(policy==policy_vanilla):
        print("Policies match")
    else:
        print("Policies do not match")
    

    
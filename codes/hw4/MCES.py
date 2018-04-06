import numpy as np

def MCES(get_episode,initial_Q,initial_policy,gamma,alpha,num_episodes=1e4):
    # This function implements the Monte Carlo ES algorithm. 
    # It returns the learned Q values and the greedy policy w.r.t. Q.
    
    # If alpha = 0, update Q[s,a] += (G - Q[s,a]) / N_sa[s,a];
    # otherwise, Q[s,a] += (G - Q[s,a]) * alpha

    # initialization  
    Q = np.copy(initial_Q)
    policy = np.copy(initial_policy)
    num_states, num_actions = Q.shape
    N_sa = np.zeros([num_states,num_actions]) #counter of (s,a)
    
    iteration = 0
    
    while iteration < num_episodes:
        iteration += 1                                        

        initialState = np.random.randint(num_states)
        initialAction = np.random.randint(num_actions)

        for ep in range(num_episodes):
            states,actions,rewards = get_episode(policy, initialState, initialAction) # generate an episode
            G_total = 0
            for s,state in enumerate(states):
                first_occr = next(i for i,x in enumerate(states) if states[i] == state )
                G = sum([ x * (gamma ** i) for i,x in enumerate(rewards[first_occr:])])
                G_total += G
                N_sa[state,actions[s]] += 1.0

                if alpha ==0:
                    Q[state,actions[s]]  += ( G_total - Q[state,actions[s]] )/N_sa[state,actions[s]] 

                else:
                    Q[state,actions[s]]  += alpha *(G_total - Q[state,actions[s]] )
        
            for state in states:
                bestAction = np.argmax(Q[state,:])
                policy[state] = Q[state,bestAction]
    return Q , policy


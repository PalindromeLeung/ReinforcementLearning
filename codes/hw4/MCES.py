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

        """
        Your code
        """
        
        iteration += 1                                        
        
    return Q , policy


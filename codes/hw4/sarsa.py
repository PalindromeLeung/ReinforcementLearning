import numpy as np

def sarsa(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, epsilon=0.1):
    """
    This function implements Sarsa. It returns learned Q values.
    To crete Figure 6.3 and 6.4, the function also returns number of steps, and 
    the total rewards in each episode.
        
    Notes on inputs:    
    -transition: function. It takes current state s and action a as parameters 
                and returns next state s', immediate reward R, and a boolean 
                variable indicating whether s' is a terminal state. 
                (See windy_setup as an example)
    -epsilon: exploration rate as in epsilon-greedy policy
    
    """    
    
    # initialization    
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape    
       
    steps = np.zeros(num_episodes,dtype=int) # store #steps in each episode
    rewards = np.zeros(num_episodes) # store total rewards for each episode
    
    for ep in range(num_episodes):
       
       """
        Your code
        """
            
    return Q,  steps, rewards
        
    
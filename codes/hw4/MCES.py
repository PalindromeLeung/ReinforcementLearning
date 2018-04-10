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
       
        
        for ep in range(num_episodes):
            
            iteration += 1   
            
            initialState = np.random.randint(num_states)
            initialAction = np.random.randint(num_actions)
#            initialAction = np.random.binomial(1,policy[initialState,1])
                   
            states,actions,rewards = get_episode(policy,initialState,initialAction) 

            G = 0
            for i,crnState in enumerate(states): 
                frstOcc = next(s for s,x in enumerate(states) if crnState == states[s])
                Gtmp = sum([x * (gamma ** r) for r,x in enumerate(rewards[frstOcc:])])
                G += Gtmp
                N_sa[crnState,actions[i]] += 1.0
                if alpha == 0:
                    Q[crnState,actions[i]] += (G - Q[crnState,actions[i]]) / N_sa[crnState,actions[i]]
                else:
                    Q[crnState,actions[i]] += alpha * (G - Q[crnState,actions[i]])
          
            for state in states:
                actIdx = np.argmax(Q[state,:])
                policy[state,:] = 0
                policy[state,actIdx] = 1
            
  
    return Q , policy


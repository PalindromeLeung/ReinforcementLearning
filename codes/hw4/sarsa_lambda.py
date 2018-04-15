import numpy as np

def sarsa_lambda(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, lambda_, epsilon=0.1):
    """
    This function implements backward view of Sarsa(lambda). It returns learned Q values.
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
        # epsilon greedy
        uniformScl = epsilon/ num_actions
        greedyScl = (1 - epsilon) + epsilon /num_actions

        crnState = initial_state
        # set up eligibility trace
        ElgTrace = np.copy(initial_Q)

        cnt = 0
        imdRewards = 0
        while True: 
            cnt +=1 
            nxtState, imdReward, terminal = transition(crnState,crnAction)
            imdRewards += imdReward
            # if it reaches terminal state then exploration
            if terminal:
                break   

            # Take action A, observe R,S'
            # Choose A' from S' using policy derived from Q(epsilon-greedy)

            nxtActPrb = uniformScl * np.ones(num_actions,dtype = float)
            nxtActPrb[np.argmax(Q[nxtState])] = greedyScl
            nxtAction = np.random.choice(num_actions, p = nxtActPrb)

            # delta = TD error
            delta = imdReward + gamma*Q[nxtState,nxtAction] - Q[crnState,crnAction]

            # update eligibility Trace 
            ElgTrace[crnState,crnAction] += 1

            for i in range(num_states):
                for j in range(num_actions):
                    Q[i,j] += Q[i,j] + alpha*delta*ElgTrace[i,j]
                    ElgTrace[i,j] = gamma*lambda_*ElgTrace[i,j]

            crnState = nxtState
            crnAction = nxtAction
        rewards[ep] += imdRewards
        steps[ep] += cnt
        
    return Q,  steps, rewards
        
    
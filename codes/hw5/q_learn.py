import numpy as np

def q_learn(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, epsilon=0.1):
              
    """
    This function implements Q-learning. It returns learned Q values.
    To crete 6.4, the function also returns number of steps, and 
    the total rewards in each episode.
        
    Notes on inputs:    
    -transition: function. It takes current state s and action a as parameters 
                and returns next state s', immediate reward R, and a boolean 
                variable indicating whether s' is a terminal state. 
                (See windy_setup as an example)
    -epsilon: exploration rate as in epsilon-greedy policy
    
    """    
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape

    steps = np.zeros(num_episodes, dtype = int)
    rewards  = np.zeros(num_episodes,dtype = float)


    for ep in range(num_episodes):

        crnState = int(initial_state)

        cnt = 0
        imdRewards = 0.0

        while True:
            cnt += 1
            if np.random.binomial(1, epsilon) == 1:
                crnAction =  np.random.choice([i for i in range(num_actions)])
            else: 
                values_ = Q[crnState]
                crnAction = np.random.choice([act for act, vl in enumerate(values_) if vl == np.max(values_)])

            nxtState, imdReward, terminal = transition(crnState,crnAction)
            if terminal:
                break

            imdRewards += imdReward

            # Take action A, observe R,S'
            # Choose A' from S' using policy derived from Q(epsilon-greedy)

            TD_target = imdReward + gamma *  max(Q[nxtState]) 
            Q[crnState][crnAction] += alpha * (TD_target - Q[crnState][crnAction])

            crnState = nxtState

        rewards[ep] += imdRewards
        steps[ep] += cnt

    return Q,  steps, rewards
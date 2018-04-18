import numpy as np

def doubleQ(initial_Q1,initial_Q2,initial_state,transition,
           num_episodes,gamma, alpha, epsilon=0.1):
    #This function implements double Q-learning. It returns Q1, Q2 and their sum Q
    
    Q1 = np.copy(initial_Q1)
    Q2 = np.copy(initial_Q2)
    num_states, num_actions = Q1.shape
    Q = Q1 + Q2

    for ep in range(num_episodes):
        crnState = initial_state

        cnt = 0
        imdRewards = 0.0

        while True:
            cnt +=1 

            uniformScl = epsilon / num_actions
            greedyScl = (1 - epsilon) + epsilon / num_actions
            actionPrb = uniformScl * np.ones(num_actions,dtype= float)
            actionPrb[np.argmax(Q[crnState,:])] = greedyScl
            crnAction = np.random.choice(num_actions,p = actionPrb)
            
            
            nxtState,imdReward, terminal = transition(crnState, crnAction)

            if terminal:
                break

            if np.random.random() > 0.5:  # representaion of 0.5 probability
                Q1[crnState,crnAction] += alpha * (imdReward + gamma *(Q2[nxtState,np.argmax(Q1[nxtState,:])]) - Q1[crnState,crnAction] )
            
            else:
                Q2[crnState,crnAction] += alpha * (imdReward + gamma *(Q1[nxtState,np.argmax(Q2[nxtState,:])]) - Q2[crnState,crnAction] )

            Q = Q1 + Q2
            crnState = nxtState

    return Q1, Q2, Q 
           
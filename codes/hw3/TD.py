import numpy as np

def TD0(get_episode,policy, initial_v, gamma, alpha,num_episodes = 1):
# This function implements TD(0).
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v

    # initialization  
    num_states = policy.shape[0]
    v = np.copy(initial_v)
    N_s = np.zeros(num_states) # counter for states
   
    for ep in range(num_episodes):
        states,_,rewards = get_episode(policy) # generate an episode

        G_total = 0
        for j,state in enumerate(states):
            N_s[state] += 1.0

            first_occr = next(i for i,x in enumerate(states) if states[i] == state )
            # Calculate G_total
            G = sum([ x * (gamma ** i) for i,x in enumerate(rewards[first_occr:])])
            G_total += G

            if alpha ==0:
                v[state] +=( G_total - v[state])/N_s[state]

            else:
                v[state] += alpha *(G_total - v[state])
    return v


def TD_n(get_episode, policy, initial_v, n, gamma, alpha,num_episodes = 1):
# This function implements n-step TD.
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# n: number of steps to look ahead
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v

    # initialization
    num_states = policy.shape[0]
    v = np.copy(initial_v)
    N_s = np.zeros(num_states) # counter for states
   
    for ep in range(num_episodes):
        states,_,rewards = get_episode(policy) # generate an episode
        G_total = 0
        for j,state in enumerate(states):
            N_s[state] += 1.0

            first_occr = next(i for i,x in enumerate(states) if states[i] == state )
            # Calculate G_total: Gt to Gt+n

            endTime = min(len(states), j+n)
            G = sum([ x * (gamma ** i) for i,x in enumerate(rewards[first_occr:endTime])])
            G_total += G
            # Calculate TD error 
            if alpha ==0:
                v[state] +=( G_total - v[state])/N_s[state]

            else:
                v[state] += alpha *(G_total - v[state])

    return v


def TD_lambda(get_episode, policy, initial_v, lambda_, gamma, alpha,
              num_episodes=1):
# This function implements n-step TD.
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# lambda_: value of lambda in TD(lambda)
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v
              
    # initialization 
    v = np.copy(initial_v)    
    num_states = policy[0]

    for ep in range(num_episodes):
        states,_,rewards = get_episode(policy)
        eTrace = np.copy(initial_v)
        for i in range(len(rewards)):
            currentState = states[i]
            eTrace *= gamma * lambda_ 
            eTrace[currentState] += 1
            delta = rewards[i] +  gamma * v[states[i+1]] - v[states[i]]
            v += alpha * delta * eTrace
    return v


        
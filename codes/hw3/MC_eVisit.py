import numpy as np

def MC_eVisit(get_episode,policy,initial_v,gamma,alpha,num_episodes=1):
# This function implements the Monte Carlo every-visit algorithm.
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# gamma: discount factor
# alpha: if alpha = 0 , updata v[s] by v[s] += (G - v[s]) / N_s[s] ;   
        # Otherwise, update v[s] by v[s] += alpha * (G - v[s])
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v

    # initialization  
    num_states = policy.shape[0]
    v = np.copy(initial_v)
    N_s = np.zeros(num_states) # counter for states
   
    for ep in range(num_episodes):
        states,_,rewards = get_episode(policy) # generate an episode
        G_total = 0
        for state in states:
            first_occr = next(i for i,x in enumerate(states) if states[i] == state )
            G = sum([ x * (gamma ** i) for i,x in enumerate(rewards[first_occr:])])
            G_total += G
            N_s[state] += 1.0

            if alpha ==0:
                v[state] +=( G_total - v[state])/N_s[state]

            else:
                v[state] += alpha *(G_total - v[state])
    return v

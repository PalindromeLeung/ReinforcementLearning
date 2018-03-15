import numpy as np

def TD0(get_episode,policy, initial_v, gamma, alpha,num_episodes = 1):
# This function implements TD(0).
# get_episode: function to generate an episode
# policy: the policy to be evaluated 
# initial_v: initial estimate for value function v
# gamma: discount factor
# alpha: learning rate
# num_episodes: number of episodes (iterations)
# The function returns the estimate of v

    # initialization  
    v = np.copy(initial_v)
    
    """ 
    Your Code
    """
    
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
    v = np.copy(initial_v)
    
    """
    Your Code
    """

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
    
    """
    Your Code
    """

    return v


        
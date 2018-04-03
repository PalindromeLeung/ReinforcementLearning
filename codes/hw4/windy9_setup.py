import numpy as np

stateSpace = np.zeros([70,2], dtype=int)
stateSpace[:,0] = np.tile(range(10),7)
stateSpace[:,1] = np.tile(np.repeat(range(7),10),1)
stateSpace = stateSpace.tolist()
terminal_state = [7,3]

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

def transition(state,action):
    """
    Your code
    """    
    return next_state, reward, terminal
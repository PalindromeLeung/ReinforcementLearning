# Reinforcement Learning 
# Homework 1 
# author@ Hannah Leung

'''
7. Write a function in Python that can generate an 
episode for the student MRP example. Each episode 
should include states and rewards. Use your function 
to generate three episodes, and append the results 
as comment following your code. Submit your code on Collab.
'''
import random
from numpy.random import choice
states = ['Class1','Class2','Class3','Pass','Pub','Facebook','Sleep']

states_dic = {0:'Class1', 1:'Class2',
			2:'Class3', 3:'Pass',
				4:'Pub', 5:"Facebook",
				6:"Sleep"}


PrTransMatrix =[[0	,0.5	,0	,0	,0	,0.5	,0	],
				[0	,0	,0.8	,0	,0	,0	,0.2	],
				[0	,0	,0	,0.6	,0.4	,0	,0	],
				[0	,0	,0	,0	,0	,0	,1.0	],
				[0.2	,0.4	,0.4	,0	,0	,0	,0	],
				[0.1	,0	,0	,0	,0	,0.9	,0	],
				[0	,0	,0	,0	,0	,0	,1.0	],
			   ]
Reward = {"Class1":-2, "Class2":-2, "Class3":-2,\
			 "Pass":10, "Pub":1, "Facebook":-1, "Sleep":0}

def EpiGenerator(probtransitionmatrix,eventsDict):
	sequence = [0]
	i = 0 
	while(True):
		cande =[]
		prob = []
		for j in range(7):
			if probtransitionmatrix[i][j]!=0:
				prob.append(probtransitionmatrix[i][j])
				cande.append(j)
		draw = choice(cande, 1, p=prob)
		i = draw[0]
		sequence.append(i)
		# i = i + 1
		if i >= 6:
			break
	return [eventsDict[x] for x in sequence]

# for generating sequences required in Problem 7
for i in range(3):
	print("=======sequence {} ==========".format(i+1))
	sequence = EpiGenerator(PrTransMatrix,states_dic)
	state_reward = []
	for j in range(len(sequence)):
		state_reward.append([sequence[j], Reward[sequence[j]]])
	print(state_reward)


# for generating sequnces required in Problem 8
'''
8Write a function in Python that can generate an episode 
for the student MDP example under a given policy. Each 
episode should include states, actions, and rewards. Then 
use your function to generate three episodes under the 
random policy. Paste results as comment in your code. Submit 
your code on Collab.
Note: for simplicity, we will assume that there are only two 
possible actions: Study and Relax (i.e., Facebook, Pub, or Sleep).
'''
StudyAction = ['Class1','Class2','Class3']
RelaxAction = ['Facebook','Pub','Sleep']

for i in range(3):
	print("=======sequence {} ==========".format(i+1))
	sequence = EpiGenerator(PrTransMatrix, states_dic)
	state_action_reward = []
	for j in range(len(sequence)):
		state_action_reward.append([sequence[j], "Study" if sequence[j] in StudyAction else "Relax", Reward[sequence[j]]])
	print(state_action_reward)
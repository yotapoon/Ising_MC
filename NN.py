import sklearn
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pickle

N = 32
beta_start = 0.24
beta_end = 0.54
step = 16
beta_list = np.linspace(beta_start, beta_end, 16)
beta = beta_list[0]
T = 200000

class Ising:
	def __init__(self,N,beta,dimension = 2,H = 0):#initialize spin
		self.spin = 2*np.random.randint(0,2,[N]*dimension)-1
		self.H = H#magnetic field from outside
		self.neighbors = np.zeros((dimension*2,dimension),dtype = "int")
		self.dimension = dimension
		self.beta = beta
		self.size = self.spin.size
		for d in range(2*dimension):
			self.neighbors[d][d//2] = (-1)**d
		self.m = self.cal_m()
	
	def initialize_configuration():
		self.spin = 2*np.random.randint(0,2,[N]*dimension)-1
		self.m = self.cal_m()
		
	def cal_m(self):#calculate magnetization
		return self.spin.sum()/self.spin.size
	
	def flip(self):#flip random cite
		cite = np.random.randint(0,N,self.dimension)#choice the cite to flip (or not)
		#Note that this specification must be conducted by tuple
		s_trial = -self.spin[tuple(cite)]#flip
		deltaE = -2.0*self.H*s_trial
		for neighbor in self.neighbors:
			cite_temp = (cite+neighbor+N)%N
			deltaE += -2.0*s_trial*self.spin[tuple(cite_temp)]
		p = min(1.0,np.exp(-self.beta*deltaE))
		if np.random.rand() < p:
			self.spin[tuple(cite)] = s_trial
			self.m += s_trial*2.0/(self.size)
		return deltaE

model = Ising(N,beta,dimension = 2)

sample_end = 5000
configurations = [[] for s in range(sample_end)]
labels = []
label = 0

for s in range(sample_end):
	print(s,end = "\r")
	label = np.random.choice(len(beta_list))
	beta = beta_list[label]
	labels.append(label)
	model = Ising(N,beta,dimension = 2)
	for t in range(T):
		model.flip()
	configurations[s] = model.spin.reshape(model.size)
data = {}
data["labels"] = np.array(labels)
data["configurations"] = np.array(configurations)

pkl_file = "data.pkl"
with open(pkl_file,"wb") as f:
	pickle.dump(data,f,-1)
"""

pkl_file = "data.pkl"
with open(pkl_file,"rb") as f:
    data = pickle.load(f)
labels = data["labels"]
configurations = data["configurations"]
"""

model = MLPClassifier(hidden_layer_sizes = (128,32,4,1))
model.fit(configurations,labels)
print(model.score(configurations,labels))
print(model.intercepts_[4])

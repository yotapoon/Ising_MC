import numpy as np
import matplotlib.pyplot as plt

N = 30
beta = 0.4
T = 800000

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
	
	def initialize_configration():
		self.spin = 2*np.random.randint(0,2,[N]*dimension)-1
		
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

m_list = []

for t in range(T):
	model.flip()
	if t%100 == 0:
		m_list.append(model.m)

plt.plot(np.arange(len(m_list)),m_list)
plt.show()

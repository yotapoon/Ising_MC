import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev


N = 32

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
	
	def initialize_configuration(self):
		self.spin = 2*np.random.randint(0,2,[N]*self.dimension)-1
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
	

beta_start = 0.3
dbeta = 0.003
beta_end = 0.5
T = 1000000
beta = beta_start
counter = 1
counter_end = 10

data = []

while beta < beta_end:
	counter = 1
	data_temp = []
	model = Ising(N,beta,dimension = 2)
	for counter in range(counter_end):
		print(counter,end = "\r")
		m_list = []
		model.initialize_configuration()
		for t in range(T):
			model.flip()
			if t %100 == 0:
				m_list.append(model.m)
		length = len(m_list)
		data_temp.append(abs(sum(m_list[int(0.6*length):length])/len(m_list[int(0.6*length):length])))
	data.append([beta,mean(data_temp),stdev(data_temp)])
	print([beta,mean(data_temp),stdev(data_temp)])
	beta += dbeta

data = np.array(data)
plt.plot(data[:,0],data[:,1],marker = ".")
plt.show()
name = "phase(N="+str(N)+").txt"
np.savetxt(name,data)

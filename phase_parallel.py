
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev
import sys
args = sys.argv
beta = float(args[1])
print(beta)

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
        return self.spin.sum()/float(self.spin.size)
    
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
            self.m += s_trial*2.0/float(self.size)
        return deltaE
    

T = 2000000
counter = 1
counter_end = 10

data_temp = []
model = Ising(N,beta,dimension = 2)
for counter in range(counter_end):
    print(counter)
    m_list = []
    model.initialize_configuration()
    for t in range(T):
        model.flip()
        if t % 1000 == 0:
            m_list.append(model.m)
    length = len(m_list)
    data_temp.append(abs(sum(m_list[int(0.6*length):length])/len(m_list[int(0.6*length):length])))
print(data_temp)

print([beta,mean(data_temp),stdev(data_temp)])
name = "phase(beta="+str(beta)+").txt"
output = str(beta) + " " + str(mean(data_temp)) + " " + str(stdev(data_temp)) + "\n"
with open(name, mode = "w") as f:
    f.write(output)

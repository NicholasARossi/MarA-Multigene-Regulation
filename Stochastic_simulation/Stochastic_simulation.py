__author__ = 'Nicholas Rossi'
### This script generates the data to be analyzed by stochastic_simulation_graphing
import numpy as np
import pandas as pd
import time
start = time.time()
# def mary_evolution(tend=2,dt=0.0001,initials):
labels=['Input','A','B','C','D','E','Control','Control2']
output_dir='downstream_data/'

tBegin = 0
tEnd = 100000
dt = 1.0

t = np.arange(tBegin, tEnd, dt)
N = t.size
initial = 0
theta = 0.0532
mu = 0
sigma = 0.35
kappa=0.139

# initial 0.621

master_value=1
beta=1
downstream_basal=1
alpha_Input=master_value

### change this value if you want to vary the promoter strength
#alphas=np.logspace(1,2,5)
alphas=[1,1,1,1,1]
alpha_A=alphas[0]
alpha_B=alphas[1]
alpha_C=alphas[2]
alpha_D=alphas[3]
alpha_E=alphas[4]


### change this value if you want to vary the hill coefficients
ns=np.linspace(1,5,5)
#ns=[2,2,2,2,2]
n1=ns[0]
n2=ns[1]
n3=ns[2]
n4=ns[3]
n5=ns[4]


### change this value if you want to vary Kds
#ks=np.logspace(2.5,4.5,5)
ks=[10000,10000,10000,10000,10000]
AK=ks[0]
BK=ks[1]

CK=ks[2]
DK=ks[3]
EK=ks[4]




#These are the calculations for the intrinisic noise multipliers for genes A and C
tau_int=5.0/np.log(2)
i_std=np.sqrt(downstream_basal+alpha_A)
c=2.0*np.power(i_std,2.0)/tau_int

i_mu=np.exp(-dt/tau_int)


sig_d=np.sqrt(0.5*c*tau_int*(1.0-np.power(i_mu,2.0)))


#quick calculation for the intrinsic noise of for gene B
i_input_std=np.sqrt(alpha_B)
c=2.0*np.power(i_input_std,2.0)/tau_int

sig_b=np.sqrt(0.5*c*tau_int*(1.0-np.power(i_mu,2.0)))

#now we're calculating the extrinsic noise
tau_cc=60.0/np.log(2)
e_std=0.35
c=2.0*np.power(e_std,2.0)/tau_cc
e_mu=np.exp(-dt/tau_cc)
sig_e=np.sqrt(0.5*c*tau_cc*(1.0-np.power(e_mu,2.0)))


IPTG_values=np.logspace(2,5,30)*beta


Input_value=[]
A_values=[]
B_values=[]
C_values=[]
D_values=[]
E_values=[]
contro_values=[]
j=0
for IPTG in IPTG_values:

    ss=IPTG/beta
    ssA=(downstream_basal+alpha_A*(((ss/AK)**n1)/(1+((ss/AK)**n1))))/beta
    ssB=(downstream_basal+alpha_B*(((ss/BK)**n2)/(1+((ss/BK)**n2))))/beta
    ssC=(downstream_basal+alpha_C*(((ss/CK)**n3)/(1+((ss/CK)**n3))))/beta
    ssD=(downstream_basal+alpha_D*(((ss/DK)**n4)/(1+((ss/DK)**n4))))/beta
    ssE=(downstream_basal+alpha_E*(((ss/EK)**n5)/(1+((ss/EK)**n5))))/beta
    ssControl=(downstream_basal)/beta
    ssControl2=(downstream_basal+alpha_A)/beta

    initials=[0,0,0,0,0,0,0,0,0,ss,ssA,ssB,ssC,ssD,ssE,ssControl,ssControl2]
    Extrinsic = np.zeros(N)
    IInput=np.zeros(N)
    IA= np.zeros(N)
    IB= np.zeros(N)
    IC= np.zeros(N)
    ID= np.zeros(N)
    IE= np.zeros(N)
    IControl=np.zeros(N)
    IControl2=np.zeros(N)

    Input=np.zeros(N)
    A=np.zeros(N)
    B=np.zeros(N)
    C=np.zeros(N)
    D=np.zeros(N)
    E=np.zeros(N)
    Control=np.zeros(N)
    Control2=np.zeros(N)




    Extrinsic[0] = initials[0]
    IInput[0]=initials[1]
    IA[0]=initials[2]
    IB[0]=initials[3]
    IC[0]=initials[4]
    ID[0]= initials[5]
    IE[0]= initials[6]
    IControl[0]= initials[7]
    IControl2[0]= initials[8]

    Input[0]=initials[9]
    A[0]=initials[10]
    B[0]=initials[11]
    C[0]=initials[12]
    D[0]=initials[13]
    E[0]=initials[14]
    Control[0]=initials[15]
    Control2[0]=initials[16]

    #We need to calculate our input noise values each time
    #These are the calculations for the intrinisic noise multipliers for genes A and C
    tau_int=5.0/np.log(2)
    i_std=np.sqrt(IPTG)
    c=2.0*np.power(i_std,2.0)/tau_int
    i_mu=np.exp(-dt/tau_int)


    sig_input=np.sqrt(0.5*c*tau_int*(1.0-np.power(i_mu,2.0)))
    #print(i_mu)



    data_storage=np.zeros((int(tEnd/dt),8))

    for i in xrange(1, N):


        IInput[i] = (i_mu*IInput[i-1]) + sig_input*np.random.normal(loc=0.0, scale=1.0)


        Input[i]=Input[i-1]+dt*(IPTG-beta*Input[i-1]+Extrinsic[i-1]+IInput[i-1])

        A[i]=A[i-1]+dt*(downstream_basal+alpha_A*(((Input[i-1]/AK)**n1)/(1+(Input[i-1]/AK)**n1))-beta*A[i-1]+Extrinsic[i-1]+IA[i-1])
        B[i]=B[i-1]+dt*(downstream_basal+alpha_B*(((Input[i-1]/BK)**n2)/(1+(Input[i-1]/BK)**n2))-beta*B[i-1]+Extrinsic[i-1]+IB[i-1])
        C[i]=C[i-1]+dt*(downstream_basal+alpha_C*(((Input[i-1]/CK)**n3)/(1+(Input[i-1]/CK)**n3))-beta*C[i-1]+Extrinsic[i-1]+IC[i-1])
        D[i]=D[i-1]+dt*(downstream_basal+alpha_D*(((Input[i-1]/DK)**n4)/(1+(Input[i-1]/DK)**n4))-beta*D[i-1]+Extrinsic[i-1]+ID[i-1])
        E[i]=E[i-1]+dt*(downstream_basal+alpha_E*(((Input[i-1]/EK)**n5)/(1+(Input[i-1]/EK)**n5))-beta*E[i-1]+Extrinsic[i-1]+IE[i-1])
        Control[i]=Control[i-1]+dt*(downstream_basal-beta*Control[i-1]+Extrinsic[i-1]+IControl[i-1])
        Control2[i]=Control2[i-1]+dt*(downstream_basal+alpha_A-beta*Control2[i-1]+Extrinsic[i-1]+IControl2[i-1])

    data_storage[:,0]=Input
    data_storage[:,1]=A
    data_storage[:,2]=B
    data_storage[:,3]=C
    data_storage[:,4]=D
    data_storage[:,5]=E
    data_storage[:,6]=Control
    data_storage[:,7]=Control2


    s = pd.DataFrame(data_storage, index=np.arange(0,tEnd,dt),columns=labels)

    s.to_pickle(output_dir+str(j)+'.pk1')
    end = time.time()
    print end - start
    j=j+1

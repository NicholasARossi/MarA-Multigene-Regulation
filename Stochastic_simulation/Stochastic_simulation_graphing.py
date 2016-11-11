__author__ = 'Nicholas Rossi'


import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcdefaults()

matplotlib.rcParams['pdf.fonttype'] = 42

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def lagevin_model(x,y):
    ys=(np.diff(((y))))/(np.diff((((x)))))
    return x[1:],ys

def elowitz_noise(red,green):
    return np.mean((red-green)**2)/(2*np.mean(red)*np.mean(green))

def elowitz_noise_alt(red,green):
    return (np.mean((red**2+green**2))-2*np.mean(red)*np.mean(green))/(2*np.mean(red)*np.mean(green))

def cv_func(x,noise):
    return noise/x
def norm_cov(a,b):
    return np.cov(a,b)[0,1]/(np.mean(a)*np.mean(b))
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

conditions=['0uM','10uM','20uM','30uM','40uM','50uM','60uM','70uM','80uM','90uM','100uM','500uM','1mM','1.5mM','2.0mM','5mM']

IPTG_names=['0uM','10uM','20uM','30uM','40uM','50uM','60uM','70uM','80uM','90uM','100uM','500uM','1mM','1.5mM','2.0mM','5.0mM']
#IPTG_names=['0uM','10uM','100uM','1mM']

folder='downstream_data'
data_dir=folder+'/'
names=mylistdir(folder)


input=[]
times=[]
protein1=[]
protein2=[]
protein3=[]
protein4=[]
protein5=[]
Control=[]
Control2=[]
input_means=[]
protein1_means=[]
protein2_means=[]
protein3_means=[]
protein4_means=[]
protein5_means=[]
names = sorted(names, key=lambda x: int(x.split('.')[0]))
for name in names:
    data = pd.read_pickle(data_dir+name)
    #input.append(data['protein'].values[500:])
    times.append(data.index.values)
    input.append(data['Input'].values)


    protein1.append(data['A'].values)
    protein1_means.append(np.mean(data['A'].values))

    protein2.append(data['B'].values)
    protein2_means.append(np.mean(data['B'].values))

    protein3.append(data['C'].values)
    protein3_means.append(np.mean(data['C'].values))

    protein4.append(data['D'].values)
    protein4_means.append(np.mean(data['D'].values))


    protein5.append(data['E'].values)
    protein5_means.append(np.mean(data['E'].values))

    Control.append(data['Control'].values)
    Control2.append(data['Control2'].values)




new_folder='simulation_graphs/'

#In order for the analytical solutions to be correct, you need to repeat the values
master_value=1
beta=1
downstream_basal=1
alpha_Input=master_value
#alphas=np.logspace(1,2,5)
#alphas=[1,2,3,4,5]
alphas=[1,1,1,1,1]
#alphas=[1,2,3,4,5]
#alphas=[4.5,4.5,4.5,4.5,4.5]
alpha_A=alphas[0]
alpha_B=alphas[1]
alpha_C=alphas[2]
alpha_D=alphas[3]
alpha_E=alphas[4]
c=1.0
ns=np.linspace(1,5,5)
#ns=[2,2,2,2,2]
n1=ns[0]
n2=ns[1]
n3=ns[2]
n4=ns[3]
n5=ns[4]

#ks=np.logspace(2.5,4.5,5)
ks=[10000,10000,10000,10000,10000]
AK=ks[0]
BK=ks[1]

CK=ks[2]
DK=ks[3]
EK=ks[4]





IPTG_cond_vect=np.logspace(2,5,30)




x_space=np.linspace(min(IPTG_cond_vect),max(IPTG_cond_vect),10000)


ss=x_space

input_means=[]
input_errors=[]

for inp in input:
    input_means.append(np.mean(inp))
    input_errors.append(np.std(inp))


plt.plot(x_space,ss,color='orange',linewidth=3)
plt.errorbar(IPTG_cond_vect,input_means,yerr=input_errors,fmt='o',color='blue')

plt.title('Input Activator as a function of IPTG')
plt.xlabel('IPTG concentration')
plt.ylabel('Steady State Activator Concentration')
plt.xscale('log')
plt.legend(['Exact Analytical Solution','Comutational Simulation'],loc='upper left')
plt.savefig((new_folder+'input_means'+'.png'))






#Now we're going to plot all the downstream gene responses:
fig1,(ax1,ax2)=plt.subplots(2,1,figsize=(4,6))
#plt.subplots_adjust(wspace=0.25, hspace=0.5)
ssA=(downstream_basal+alpha_A*(((ss/AK)**n1)/(1+((ss/AK)**n1))))/beta
ssB=(downstream_basal+alpha_B*(((ss/BK)**n2)/(1+((ss/BK)**n2))))/beta
ssC=(downstream_basal+alpha_C*(((ss/CK)**n3)/(1+((ss/CK)**n3))))/beta
ssD=(downstream_basal+alpha_D*(((ss/DK)**n4)/(1+((ss/DK)**n4))))/beta
ssE=(downstream_basal+alpha_E*(((ss/EK)**n5)/(1+((ss/EK)**n5))))/beta
ssCon=((downstream_basal)/beta)*np.ones(len(ss))
ssCon2=((downstream_basal+alpha_A)/beta)*np.ones(len(ss))
downstreams=[protein1,protein2,protein3,protein4,protein5,Control,Control2]
downstreams_noise=[protein1,protein2,protein3,protein4,protein5]

exacts=[ssA,ssB,ssC,ssD,ssE,ssCon,ssCon2]

ssx=np.logspace(1,6,num=10000)
ssA=(downstream_basal+alpha_A*(((ssx/AK)**n1)/(1+((ssx/AK)**n1))))/beta
ssB=(downstream_basal+alpha_B*(((ssx/BK)**n2)/(1+((ssx/BK)**n2))))/beta
ssC=(downstream_basal+alpha_C*(((ssx/CK)**n3)/(1+((ssx/CK)**n3))))/beta
ssD=(downstream_basal+alpha_D*(((ssx/DK)**n4)/(1+((ssx/DK)**n4))))/beta
ssE=(downstream_basal+alpha_E*(((ssx/EK)**n5)/(1+((ssx/EK)**n5))))/beta
ssCon=((downstream_basal)/beta)*np.ones(len(ssx))
ssCon2=((downstream_basal+alpha_A)/beta)*np.ones(len(ssx))
grey_exacts=[ssA,ssB,ssC,ssD,ssE,ssCon,ssCon2]

#I could just take the derivative of these above values, but I think i'm going to calculated the analytical derivative - it may work better

ssNA=(alpha_A/beta)*(n1*ss**(n1-1)*AK**(-n1))/((ss/AK)**n1+1)**2
ssNB=(alpha_B/beta)*(n2*ss**(n2-1)*BK**(-n2))/((ss/BK)**n2+1)**2
ssNC=(alpha_C/beta)*(n3*ss**(n3-1)*CK**(-n3))/((ss/CK)**n3+1)**2
ssND=(alpha_D/beta)*(n4*ss**(n4-1)*DK**(-n4))/((ss/DK)**n4+1)**2
ssNE=(alpha_E/beta)*(n5*ss**(n5-1)*EK**(-n5))/((ss/EK)**n5+1)**2

ssNcon=(alpha_E/beta)*np.ones(len(ss))

exact_noise=[ssNA,ssNB,ssNC,ssND,ssNE,np.zeros(len(ssNE))]

ssNA=(alpha_A/beta)*(n1*ssx**(n1-1)*AK**(-n1))/((ssx/AK)**n1+1)**2
ssNB=(alpha_B/beta)*(n2*ssx**(n2-1)*BK**(-n2))/((ssx/BK)**n2+1)**2
ssNC=(alpha_C/beta)*(n3*ssx**(n3-1)*CK**(-n3))/((ssx/CK)**n3+1)**2
ssND=(alpha_D/beta)*(n4*ssx**(n4-1)*DK**(-n4))/((ssx/DK)**n4+1)**2
ssNE=(alpha_E/beta)*(n5*ssx**(n5-1)*EK**(-n5))/((ssx/EK)**n5+1)**2


grey_exact_noise=[ssNA,ssNB,ssNC,ssND,ssNE,np.zeros(len(ssNE))]


colors=cm.rainbow(np.linspace(0, 1, 6))

recs=[]
l=0
noise_terms=[]

for downstream in downstreams[0:6]:
    temp_means=[]
    temp_stds=[]
    for point in downstream:
        temp_means.append(np.mean(point))
        temp_stds.append(np.std(point))


    recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[l]))




    ax1.plot(ssx,grey_exacts[l]-c,color='grey')
    ax1.plot(ss,exacts[l]-c,color=colors[l],linewidth=3)
    ax1.errorbar(input_means,[x - c for x in temp_means],yerr=temp_stds,fmt='o',color=colors[l])
    l=l+1
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
#ax1.legend(recs,['n=1','n=2','n=3','n=4','n=5','Control'],title='Hill coefficient (n)',ncol=3,loc='upper left',bbox_to_anchor=(0, -.1))
#ax1.set_title('Noise responses\n n variation')
#ax1.legend(recs,['$10^2$','$10^{2.75}$','$10^{3.5}$','$10^{4.25}$','$10^5$','Control'],title='Dissociation constant',ncol=3,loc='upper left',bbox_to_anchor=(0, -.1))
#ax1.set_title('Comparative downstream response\n $K_d$ variation')
#ax1.set_title('Comparative downstream response\n'+ r'$\alpha$ variation')

ax1.set_xscale('log')
ax1.set_xlabel('Input Concentration')
ax1.set_ylabel('Output Concentration')
#plt.savefig(new_folder+'n_activation'+'.pdf',bbox_inches='tight')


#fig2,(ax2)=plt.subplots(1,1)
for y,downstream in enumerate(downstreams[0:6]):
    temp_noise=[]
    temp_stds=[]
    for z,point in enumerate(downstream):
        super_temp_noise=[]
        for _ in range(100):
            indxz=np.random.uniform(1,len(point),100).astype(int)
            super_temp_noise=np.append(super_temp_noise,(np.std(point[indxz])/np.mean(point[indxz]))/(np.std(input[z][indxz])/np.mean(input[z][indxz])))
        temp_noise.append(np.mean(super_temp_noise))
        temp_stds.append(np.std(super_temp_noise))



    ys=grey_exact_noise[y]/grey_exacts[y]*ssx
    plt.plot(ssx,ys,color='grey')
    #
    #
    # #plot color over the the range relevent to the emperical values
    xs=ss
    ys=exact_noise[y]/exacts[y]*ss
    ax2.plot(xs[np.logical_and(xs>min(input_means), xs<max(input_means))],(ys[np.logical_and(xs>min(input_means), xs<max(input_means))]),color=colors[y],linewidth=3)


    ax2.errorbar(input_means,temp_noise,yerr=temp_stds,fmt='o',color=colors[y])


    l=l+1
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
plt.legend(recs,['n=1','n=2','n=3','n=4','n=5','Control'],title='Hill coefficient (n)',ncol=3,loc='upper left',bbox_to_anchor=(0, -.1))
# plt.title('Noise responses\n n variation')
#plt.legend(recs,[r'$\alpha$=1',r'$\alpha$=2',r'$\alpha$=3',r'$\alpha$=4',r'$\alpha$=5','Control'],title=r'Promoter Activity ($\alpha$)',ncol=3,loc='upper left',bbox_to_anchor=(-0.25, -0.2))
#plt.legend(recs,['10$^{2.5}$','10$^3$','10$^{3.5}$','10$^4$','10$^{4.5}$','Control'],title='Dissociation constant ($K_d$)',ncol=3,loc='upper left',bbox_to_anchor=(-0.25, -0.2))
#plt.title('Noise responses')
plt.xscale('log')
plt.xlabel('Input Concentration')
plt.ylabel('Transmitted noise')
ax1.set_ylim([-.1,1.2])
ax2.set_ylim([-.1,1])
plt.savefig(new_folder+'combined'+'.pdf',bbox_inches='tight')

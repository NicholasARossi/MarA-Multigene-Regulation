__author__ = 'Nicholas Rossi'
# Written for Python 2.7
# import packages
import os
import scipy.io as sio
import matplotlib
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def hill(x,a,b,c,d):
    return a * ((np.power((x/d),b))/(1+(np.power((x/d),b))))+ c

def noise_func(x,a,b,d):
    return a*(b*d**(-b)*x**(b-1))/((x/d)**b+1)**2

def diff_hill(parameters, *data):
    a,b,c,d,s=parameters

    x,y,yn=data
    result = 0

    for i in range(len(x)):
        result += ((hill(x[i],a,b,c,d)-y[i])/y[i])**2+((s+noise_func(x[i],a,b,d)*x[i]/hill(x[i],a,b,c,d)-yn[i])/yn[i])**2
#1000000

    return result

def complete(x,a,n,c,d):
    return ((noise_func(x,a,n,d)**2)/(hill(x,a,n,c,d)+1000*x*(noise_func(x,a,n,d)**2)))**(.5)



def Total_graphage(stressors,stress_titles,meta_inputs,meta_outputs,titles,nbins,best_fit=True,normalized=False):
    import scipy.integrate as integrate
    import seaborn as sns
    sns.set_style('ticks')
    import matplotlib
    matplotlib.rcdefaults()
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    from matplotlib import gridspec
    matplotlib.rcParams['pdf.fonttype'] = 42
    # sns.set_style('white')
    #### Here we initiate colors and global limits to be used throughout
    channel_cappacities=[]
    WT=np.load('numpy_files/WT.npy')
    MV=np.load('numpy_files/MV.npy')
    n_max=1000
    #strain colors
    strain_colors = cm.rainbow(np.linspace(0, 1, len(stressors[0])))
    #stress colors


    stress_colors = [(236,95,103,255),(95,179,179,255),(197,148,197,255),(249,145,87,255),(153,199,148,255),(250,200,99,255),(102,153,204,255),(171,121,103,255),(97,124,141,255),(52,61,70,255)]
    chimeric_titles=[]
    for i in range(len(stress_colors)):
        r, g, b, o = stress_colors[i]
        stress_colors[i] = (r / 255., g / 255., b / 255., o / 255.)
    y_lims=[500,20000]

    strain_colors=strain_colors.tolist()+stress_colors[7:]
    strain_colors[1], strain_colors[0] = strain_colors[0], strain_colors[1]
    strain_colors[4], strain_colors[2] = strain_colors[2], strain_colors[4]
    #### This section creates the background heatmaps for figure 4
    fig4, (ax41,ax42) = plt.subplots(1,2)

    fig4.set_size_inches(15, 5)

    space_array=np.zeros((100,100))
    space_array2=np.zeros((100,100))
    #x is the location on the axis, y is the kd

    ns=np.linspace(0.5,5,100)
    kds=np.logspace(2,5,100)

    save_loc='result_graphs/'
    WT_levels=np.linspace(np.percentile(WT, 5), np.percentile(WT, 95),100)
    MV_levels=np.linspace(np.percentile(MV, 5), np.percentile(MV, 95),100)

    WT_levs=np.linspace(np.percentile(WT, 5), np.percentile(WT, 95),100)
    MV_levs=np.linspace(np.percentile(MV, 5), np.percentile(MV, 95),100)

    for l in range(100):
        for m in range(100):

            space_array[l,m]=np.log2(integrate.quad(lambda x: complete(x,1,ns[m],0,kds[l]), WT_levs[0], WT_levs[-1])[0])+np.log2(((n_max)/(2*np.pi*np.e))**.5)
            space_array2[l,m]=np.log2(integrate.quad(lambda x: complete(x,1,ns[m],0,kds[l]),MV_levs[0], MV_levs[-1])[0])+np.log2(((n_max)/(2*np.pi*np.e))**.5)
    space_array[np.isnan(space_array)]=0
    kdloc,nloc=np.where(space_array==np.nanmax(space_array))

    print('max n is : '+str(ns[nloc[0]]))
    print('max kd is : '+str(kds[kdloc[0]]))

    yticks=kds
    keptticks = yticks[::int(len(yticks)/3)]
    yticks = ['' for y in yticks]
    #yticks[::int(len(yticks)/3) ]= np.around(keptticks,decimals=-1)
    yticks[::int(len(yticks)/3) ]= ['10$^2$','10$^3$','10$^4$','10$^5$']

    xticks=ns
    keptticks = xticks[::int(len(xticks)/3)]
    xticks = ['' for y in xticks]
    xticks[::int(len(xticks)/3) ]= np.around(keptticks,decimals=1)


    first_plot=sns.heatmap(np.flipud(space_array),yticklabels=yticks[::-1],vmin=0,vmax=3,xticklabels=xticks,cmap="jet",rasterized=True,linewidths=0,ax=ax41)
    for item in first_plot.get_yticklabels():
        item.set_rotation(0)

    second_plot=sns.heatmap(np.flipud(space_array2),yticklabels=yticks[::-1],vmin=0,vmax=3,xticklabels=xticks,cmap="jet",linewidths=0, rasterized=True,ax=ax42)
    for item in second_plot.get_yticklabels():
        item.set_rotation(0)
    ax41.set_xlabel('Hill Coefficient (n)')
    ax41.set_ylabel('Dissociation Constant ($K_d$)')
    ax42.set_xlabel('Hill Coefficient (n)')
    ax42.set_ylabel('Dissociation Constant ($K_d$)')
    ax41.tick_params(axis=u'both', which=u'both',length=0)
    ax42.tick_params(axis=u'both', which=u'both',length=0)


    ### Heatmaps for figure 5
    fig5, (ax51,ax52) = plt.subplots(1,2)
    fig5.set_size_inches(15, 5)
    first_plot=sns.heatmap(np.flipud(space_array),yticklabels=yticks[::-1],xticklabels=xticks,cmap="jet",vmin=0,vmax=3.0, rasterized=True,linewidths=0,ax=ax51)
    for item in first_plot.get_yticklabels():
        item.set_rotation(0)

    second_plot=sns.heatmap(np.flipud(space_array2),yticklabels=yticks[::-1],xticklabels=xticks,cmap="jet",vmin=0,vmax=3.0,linewidths=0, rasterized=True,ax=ax52)
    for item in second_plot.get_yticklabels():
        item.set_rotation(0)
    ax51.set_xlabel('Hill Coefficient (n)')
    ax51.set_ylabel('Dissociation Constant ($K_d$)')
    ax52.set_xlabel('Hill Coefficient (n)')
    ax52.set_ylabel('Dissociation Constant ($K_d$)')
    ax51.tick_params(axis=u'both', which=u'both',length=0)
    ax52.tick_params(axis=u'both', which=u'both',length=0)

    #Here we set a series of figure names:
    save_loc='output_graphs/'
    fig3, (ax31,ax32) = plt.subplots(2,1,figsize=(4,6))
    fig5a, (ax5a1,ax5a2) = plt.subplots(2,1,figsize=(4,6))
    #fig4a_schematic = gridspec.GridSpec(2, 2, height_ratios=[3, 1],width_ratios=[3,1])
    fig4c_mapped_dist,((ax4c1,ax4c2),(ax4c3,ax4c4)) = plt.subplots(2, 2, gridspec_kw ={'height_ratios':[3, 1],'width_ratios':[3,1]})
    fig4d_mapped_dist,((ax4d1,ax4d2),(ax4d3,ax4d4)) = plt.subplots(2, 2, gridspec_kw ={'height_ratios':[3, 1],'width_ratios':[3,1]})
    ax4c2_2=ax4c2.twiny()




    #here we create the control plot
    Scontrol, figscontrol  = plt.subplots(1,1)
    #here we set up the mapped distributions
    s = WT
    #plt.hist(s,bins=np.logspace(0, 4, 100),normed=True,histtype='step',color='black',linewidth=3)
    #sns.distplot(s,hist=False,kde_kws={"color":'grey',"shade":True, "lw": 3},ax=ax4c3)
    weights = np.ones_like(s)/len(s)
    ax4c3.hist(s, bins=np.logspace(2,5,100),weights=weights,histtype='step',alpha=0.75,linewidth=3,color='grey')
    ax4c3.set_xscale('log')

    s2 = MV
    #plt.hist(s,bins=np.logspace(0, 4, 100),normed=True,histtype='step',color='black',linewidth=3)
    #sns.distplot(s2,hist=False,kde_kws={"color":'grey',"shade":True, "lw": 3},ax=ax4d3)
    weights = np.ones_like(s2)/len(s2)
    ax4d3.hist(s2, bins=np.logspace(2,5,100),weights=weights,histtype='step',alpha=0.75,linewidth=3,color='grey')
    ax4d3.set_xscale('log')


    #First we need to create the background heat maps that will be populated by cells and stuff
    recs1=[]
    recs2=[]
    recs3=[]

    for k,inputs in enumerate(meta_inputs):
        outputs=meta_outputs[k]
        fig=plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 2],height_ratios=[4, 4])
        ax1 = plt.subplot(gs[1])
        stress_recs=[]
        if k<6:
            stresses=stressors[k]
            for j,stress in enumerate(stresses):
                weights = np.ones_like(stress[0])/len(stress[0])
                line=ax1.hist(stress[0], bins=np.logspace(2,4.2,100),weights=weights,histtype='step',linewidth=3,color=stress_colors[j],orientation="horizontal")
                stress_recs.append(mpatches.Rectangle((0,0),1,1,fc=stress_colors[j]))

        if k==1:
            ax1.set_xticks([0,0.1,0.2])
            ax1.set_xticklabels(['0','0.1','0.2'])
        else:
            ax1.set_xticks([0,0.25,0.5])
            ax1.set_xticklabels(['0','0.25','0.5'])
        ax1.set_ylim(y_lims)
        ax2 = plt.subplot(gs[0])
        plt.xscale('log')
        plt.yscale('log')
        ax2.set_ylim(ax1.get_ylim())
        ax1.legend(stress_recs,stress_titles,bbox_to_anchor=(1.1, -.2), loc=1)


        store=nbins
        bin_min=2.8
        bin_max=4


        bins=np.logspace(bin_min,bin_max,nbins)


        #we're going to need to fill in matricies with the mean and points over certain binned inputs
        red_mean=np.zeros((100,nbins))
        green_mean=np.zeros((100,nbins))
        noise=np.zeros((100,nbins))
        norm_noise=np.zeros((100,nbins))
        green_std=np.zeros((100,nbins))
        inds=np.digitize(inputs,bins)

        #Next we need to perform some sort of operation on all the data for each of the idx
        sizes=[]
        for i in range(nbins):
            greens=outputs[inds==i]
            if len(greens)<=10:
                ### warning if there aren't enough cells in the bin
                print('warning: only ' + str(len(greens)) +' cells in bin:'+ str(bins[i])+'for sample: '+ titles[k] )

            reds=inputs[inds==i]
            sizes.append(len(greens))
            for m in range(100):
                size=int(len(greens)/3)
                indxz=np.random.uniform(1,len(greens),size).astype(int)
                red_temp=reds[indxz]
                green_temp=greens[indxz]

                red_mean[m,i]=np.mean(reds)
                green_std[m,i]=np.std(greens)
                green_mean[m,i]=np.mean(greens)

                noise[m,i]=(np.std(green_temp)/np.mean(green_temp))/(np.std(red_temp)/np.mean(red_temp))
        sizes=np.asarray(sizes)
        plt.scatter(inputs,outputs,color=strain_colors[-1],alpha=0.1)
        thresh=25
        green_mean=green_mean[0][sizes>thresh]#[0:-1]
        red_mean=red_mean[0][sizes>thresh]#[0:-1]
        green_std=green_std[0][sizes>thresh]#[0:-1]
        noise_mean=np.mean(noise,0)[sizes>thresh]#[0:-1]
        noise_std=np.std(noise,0)[sizes>thresh]#[0:-1]

        x_points=red_mean
        y_points=green_mean
        y_noise=noise_mean
        green_std=green_std
        #### Theses values are set to prevent fitting system from being stuck in local attractors.
        if best_fit==True:
            if k==1 or k==3:
                cmin=500
                cmax=10000
                nmax=4.5
                kdmin=700
                kdmax=1000
                nmin=1
            elif k==6:
                cmin=0
                cmax=10000
                nmin=3
                nmax=5
                kdmin=100
                kdmax=1000
            elif k==7:
                cmin=0
                cmax=10000
                nmax=2
                kdmin=1000
                kdmax=20000
                nmin=1
            elif k==8:
                cmin=0
                cmax=10000
                nmax=2
                kdmin=0
                kdmax=2000000
                nmin=0
            else:
                cmin=0
                cmax=10000
                nmax=1.5
                kdmin=1000
                kdmax=10000
                nmin=1
            bounds = [(0,1000000), (nmin, nmax), (cmin, cmax),(kdmin, kdmax),(0, 10000)]

            args = (x_points,y_points,y_noise)
            result = differential_evolution(lambda parameters,*data: diff_hill(parameters, *data), bounds, args=args)
            #popt, pcov = curve_fit( lambda x, a, b, d: hill(x, a, b, c,d), x_points, y_points,sigma=np.std(green_mean,0),p0=p0,maxfev=1000000)
            #print(result.x)
            popt=result.x

            print(titles[k]+" & "+str(int(popt[0])) +" & "+ str(int(popt[3])) +" & "+ str("{0:.2f}".format(popt[1]))+" & "+ str(int(popt[2]))+" & "+ str("{0:.2f}".format(popt[4])))
            x=np.logspace(2,5,1000)
            y=hill(x,popt[0],popt[1],popt[2],popt[3])
            x2=np.logspace(np.log10(min(x_points)),np.log10(max(x_points)),1000)
            y2=hill(x2,popt[0],popt[1],popt[2],popt[3])
            if normalized==True:
                y=(y-popt[2])/abs(popt[0])
                y2=(y2-popt[2])/abs(popt[0])
                y_points=(y_points-popt[2])/abs(popt[0])
                green_mean=(green_mean-popt[2])/abs(popt[0])


        #This is for the non chimeric graphs:
        if k<6:
            recs1.append(mpatches.Rectangle((0,0),1,1,fc=strain_colors[k]))
            new_n_locs=find_nearest(ns,popt[1])
            new_kds_locs=find_nearest(kds,popt[3])
            ax41.scatter(new_n_locs, new_kds_locs,facecolor=strain_colors[k],s=80)
            ax42.scatter(new_n_locs, new_kds_locs,facecolor=strain_colors[k],s=80)

            channel_cappacity_wt=np.log2(integrate.quad(lambda x: complete(x,1,popt[1],0, popt[3]), np.percentile(WT, 5), np.percentile(WT, 95))[0])+np.log2(((n_max)/(2*np.pi*np.e))**.5)
            channel_cappacity_mv = np.log2(integrate.quad(lambda x: complete(x,1,popt[1],0,popt[3]),np.percentile(MV, 5), np.percentile(MV, 95))[0])+np.log2(((n_max)/(2*np.pi*np.e))**.5)

            ax41.annotate(titles[k], (new_n_locs+1,new_kds_locs+1),color='White')
            ax42.annotate(titles[k], (new_n_locs+1,new_kds_locs+1),color='White')
            ax31.plot(x, (y-popt[2])/abs(popt[0]),color='grey')
            ax31.plot(x2, (y2-popt[2])/abs(popt[0]),color=strain_colors[k],linewidth=3)
            s_out = hill(s, 1, popt[1], 0, popt[3])
            s2_out = hill(s2, 1, popt[1], 0, popt[3])
            ### here we save the mapped distributions:
            np.save('compiled_mapped_distributions/cp_wt/'+ titles[k],channel_cappacity_wt)
            np.save('compiled_mapped_distributions/cp_mv/' + titles[k], channel_cappacity_mv)

            np.save('compiled_mapped_distributions/WT/' + titles[k], s_out)
            ### here we save the mapped distributions:
            np.save('compiled_mapped_distributions/MV/' + titles[k], s2_out)


            if k==1 or k==2:
                ax4c1.plot(x, (y-popt[2])/abs(popt[0]),color=strain_colors[k],linewidth=3,label=titles[k])
                ax4d1.plot(x, (y-popt[2])/abs(popt[0]),color=strain_colors[k],linewidth=3,label=titles[k])

                weights = np.ones_like(s_out)/len(s_out)
                if k==1:
                    ax4c2_2.hist(s_out, bins=np.linspace(0,1,50),histtype="step",weights=weights,color=strain_colors[k],edgecolor=strain_colors[k],linewidth=3,orientation="horizontal")

                else:
                    ax4c2.hist(s_out, bins=np.linspace(0,1,50),histtype="step",weights=weights,color=strain_colors[k],edgecolor=strain_colors[k],linewidth=3,orientation="horizontal")

                weights = np.ones_like(s2_out)/len(s2_out)

                ax4d2.hist(s2_out, bins=np.linspace(0,1,50),histtype="step",weights=weights,color=strain_colors[k],edgecolor=strain_colors[k],linewidth=3,orientation="horizontal")


            ax31.errorbar(x_points,(y_points-popt[2])/abs(popt[0]),yerr=green_std/abs(popt[0]),color=strain_colors[k],fmt='o')
            ax32.plot(x, (noise_func(x,popt[0],popt[1],popt[3])*x/hill(x,popt[0],popt[1],popt[2],popt[3])),color='grey')
            ax32.plot(x2, (noise_func(x2,popt[0],popt[1],popt[3])*x2/hill(x2,popt[0],popt[1],popt[2],popt[3])),color=strain_colors[k],linewidth=3)
            #this is the third plot
            ax32.errorbar(x_points,noise_mean-popt[4],yerr=noise_std,color=strain_colors[k],fmt='o')

        if k==6 or k==7 or k==0 or k==3:
            chimeric_titles.append(titles[k])
            recs2.append(mpatches.Rectangle((0,0),1,1,fc=strain_colors[k]))
            new_n_locs=find_nearest(ns,popt[1])
            new_kds_locs=find_nearest(kds,popt[3])
            ax51.scatter(new_n_locs, new_kds_locs,facecolor=strain_colors[k],s=80)
            ax52.scatter(new_n_locs, new_kds_locs,facecolor=strain_colors[k],s=80)
            ax51.annotate(titles[k], (new_n_locs+1,new_kds_locs+1),color='White')
            ax52.annotate(titles[k], (new_n_locs+1,new_kds_locs+1),color='white')
            ax5a1.plot(x, (y-popt[2])/abs(popt[0]),color='grey')
            ax5a1.plot(x2, (y2-popt[2])/abs(popt[0]),color=strain_colors[k],linewidth=3)
            ax5a1.errorbar(x_points,(y_points-popt[2])/abs(popt[0]),yerr=green_std/abs(popt[0]),color=strain_colors[k],fmt='o')
            ax5a2.plot(x, (noise_func(x,popt[0],popt[1],popt[3])*x/hill(x,popt[0],popt[1],popt[2],popt[3])),color='grey')
            ax5a2.plot(x2, (noise_func(x2,popt[0],popt[1],popt[3])*x2/hill(x2,popt[0],popt[1],popt[2],popt[3])),color=strain_colors[k],linewidth=3)
            ax5a2.errorbar(x_points,noise_mean-popt[4],yerr=noise_std,color=strain_colors[k],fmt='o')

        #here we plot control group stuff
        if k==1 or k==8:
            figscontrol.plot(x, y,color='grey',linewidth=3)
            figscontrol.plot(x2, y2,color='black',linewidth=3)
            figscontrol.errorbar(x_points,y_points,yerr=green_std,color='black',fmt='o')
            figscontrol.scatter(inputs,outputs,color=strain_colors[k],alpha=0.35)
            recs3.append(mpatches.Rectangle((0,0),1,1,fc=strain_colors[k]))

        #here we complete the individual graphs
        ax2.plot(x, y,color='grey')
        ax2.plot(x2, y2,color=strain_colors[k],linewidth=3)
        ax2.errorbar(x_points,y_points,yerr=green_std,color='white',fmt='o')
        ax3 = plt.subplot(gs[2])
        ax3.set_xscale('log')
        ax3.plot(x, (noise_func(x,popt[0],popt[1],popt[3])*x/hill(x,popt[0],popt[1],popt[2],popt[3])),color='grey')
        ax3.plot(x2, (noise_func(x2,popt[0],popt[1],popt[3])*x2/hill(x2,popt[0],popt[1],popt[2],popt[3])),color=strain_colors[k],linewidth=3)

        #this is the first plot
        ax3.errorbar(x_points,noise_mean-popt[4],yerr=noise_std,color=strain_colors[k],fmt='o')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax1.set_yscale('log')
        ax3.set_ylim([-0.5,3.5])

        ax2.set_xlim([10**2.5,10**4.2])

        plt.suptitle(titles[k])
        ax3.set_xlim(ax2.get_xlim())
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('GFP (a.u.)')
        ax2.set_ylabel('GFP (a.u.)')
        ax2.set_xlabel('RFP (a.u.)')
        ax2.set_title('Activation')
        ax3.set_title('Noise')

        ax3.set_ylabel('Transmitted Noise')
        ax3.set_xlabel('RFP (a.u.)')

        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_ticks_position('left')
        ax3.xaxis.set_ticks_position('bottom')

        plt.tight_layout()
        fig.savefig((save_loc+titles[k]+'.pdf'),bbox_inches='tight')


    #Here we enumerate the details for Figure 3:
    ax31.set_xscale('log')
    ax32.set_xscale('log')
    ax31.set_xlabel('RFP (a.u.)')
    ax32.set_xlabel('RFP (a.u.)')
    ax31.set_xlim([10**2.5,10**4.2])
    ax32.set_xlim([10**2.5,10**4.2])

    ax31.set_ylabel('Normalized GFP (a.u.)')
    ax32.set_ylabel('Transmitted Noise')
    ax32.legend(recs1,titles[0:6],title='Strains',bbox_to_anchor=(0, -0.2), loc=2,ncol=3)
    ax31.yaxis.set_ticks_position('left')
    ax31.xaxis.set_ticks_position('bottom')
    ax32.yaxis.set_ticks_position('left')
    ax32.xaxis.set_ticks_position('bottom')
    ax32.set_ylim([-0.5,3.5])
    fig3.savefig((save_loc+'trajectories.pdf'),bbox_inches='tight')

    fig4.savefig((save_loc+'heat_map.pdf'),bbox_inches='tight')
    ax4c1.set_xscale('log')
    ax4c3.set_xscale('log')
    ax4c1.set_xlim([10**2.5,10**4.2])
    ax4c3.set_xlim([10**2.5,10**4.2])
    ax4c1.set_ylim([0,1.2])
    ax4c2.set_ylim([0,1.2])
    ax4d1.set_xscale('log')
    ax4d3.set_xscale('log')
    ax4d1.set_xlim([10**2.5,10**4.2])
    ax4d3.set_xlim([10**2.5,10**4.2])
    ax4d1.set_ylim([0,1.2])
    ax4d2.set_ylim([0,1.2])


    ax4c3.set_ylabel('Probability')
    ax4c3.set_yticks([0,0.125,0.25])
    ax4c3.set_yticklabels(['0','0.125','0.25'])

    ax4c2_2.set_xticks([0,0.05,0.1])
    ax4c2_2.set_xticklabels(['0','0.05','0.1'],color=strain_colors[1])
    ax4c2.set_xticks([0,0.3,0.6])
    ax4c2.set_xticklabels(['0','0.3','0.6'],color=strain_colors[2])
    ax4c2.set_xlabel('Probability')

    ax4c2.set_ylabel('Predicted Downstream Concentration')

    ax4c3.set_xlabel('Estimated MarA Concentration')
    ax4c1.set_xlabel('Input Concentration')
    ax4c1.set_ylabel('Output Concentration')
    ax4c1.legend(title='Strain',bbox_to_anchor=(1.11, -0.2),ncol=1,loc=2)

    ax4d2.set_ylabel('Predicted Downstream Concentration')
    #ax4d3.set_yticks([0,0.001])
    ax4d3.set_ylabel('Probability')
    ax4d3.set_xlabel('Estimated MarA Concentration')
    ax4d1.set_xlabel('Input Concentration')
    ax4d1.set_ylabel('Output Concentration')
    ax4d1.legend(title='Strain',bbox_to_anchor=(1.11, -0.2),ncol=1,loc=2)
    ax4d3.set_ylabel('Probability')
    ax4d3.set_yticks([0,0.075,0.15])
    ax4d3.set_yticklabels(['0','0.075','0.15'])
    ax4d2.set_ylabel('Probability')
    ax4d2.set_xticks([0,0.125,0.25])
    ax4d2.set_xticklabels(['0','0.125','0.25'])

    ax4c1.plot(WT_levs,0.5*np.ones(len(WT_levs)),linewidth=3,color='grey')
    ax4c1.axvspan(WT_levs[0], WT_levs[-1], color='grey', alpha=0.5, lw=0)


    ax4d1.plot(MV_levs,0.5*np.ones(len(MV_levs)),linewidth=3,color='grey')
    ax4d1.axvspan(MV_levs[0], MV_levs[-1], color='grey', alpha=0.5, lw=0)

    ax4c1.yaxis.set_ticks_position('left')
    ax4c1.xaxis.set_ticks_position('bottom')
    ax4c2.yaxis.set_ticks_position('left')
    ax4c2.xaxis.set_ticks_position('bottom')
    ax4c3.yaxis.set_ticks_position('left')
    ax4c3.xaxis.set_ticks_position('bottom')

    ax4d1.yaxis.set_ticks_position('left')
    ax4d1.xaxis.set_ticks_position('bottom')
    ax4d2.yaxis.set_ticks_position('left')
    ax4d2.xaxis.set_ticks_position('bottom')
    ax4d3.yaxis.set_ticks_position('left')
    ax4d3.xaxis.set_ticks_position('bottom')

    fig4c_mapped_dist.delaxes(ax4c4)
    fig4c_mapped_dist.savefig((save_loc+'wt_mapped_dist.pdf'),bbox_inches='tight')
    fig4d_mapped_dist.delaxes(ax4d4)
    fig4d_mapped_dist.savefig((save_loc+'mv_mapped_dist.pdf'),bbox_inches='tight')

    ###Figure 5 saving
    ax5a1.set_xscale('log')
    ax5a2.set_xscale('log')
    ax5a1.set_xlim([10**2.5,10**4.2])
    ax5a2.set_xlim([10**2.5,10**4.2])
    ax5a1.set_xlabel('RFP (a.u.)')
    ax5a2.set_xlabel('RFP (a.u.)')

    ax5a1.set_ylabel('Normalized GFP (a.u.)')
    ax5a2.set_ylabel('Transmitted Noise')
    ax5a1.yaxis.set_ticks_position('left')
    ax5a1.xaxis.set_ticks_position('bottom')
    ax5a2.yaxis.set_ticks_position('left')
    ax5a2.xaxis.set_ticks_position('bottom')

    ax5a2.legend(recs2,chimeric_titles,title='Strains',bbox_to_anchor=(0, -0.2), loc=2,ncol=2)
    ax5a2.set_ylim([-0.5, 3.5])
    fig5a.savefig((save_loc+'chimera_trajectories.pdf'),bbox_inches='tight')
    fig5.savefig((save_loc+'chimera_heat_map.pdf'),bbox_inches='tight')

    figscontrol.set_ylim([500,20000])
    figscontrol.set_xlim([10**2.5,10**4.2])
    figscontrol.set_ylabel('GFP (a.u.)')
    figscontrol.set_xlabel('RFP (a.u.)')
    figscontrol.yaxis.set_ticks_position('left')
    figscontrol.xaxis.set_ticks_position('bottom')
    figscontrol.set_xscale('log')
    figscontrol.set_yscale('log')
    figscontrol.legend(recs3,['$P_{micF}$','Control'],title='Strains',loc=9, ncol=2,bbox_to_anchor=(0.5, -0.1))
    Scontrol.savefig((save_loc+'control.pdf'),bbox_inches='tight')

    return fig

if __name__ == "__main__":
    # make sure we've got white backgrounds and readable text
    matplotlib.rcdefaults()
    matplotlib.rcParams['pdf.fonttype'] = 42

    # Here we collect all of the data that we will analyze

    titles = [r'P$_{acrAB}$', r'P$_{micF}$', r'P$_{inaA}$', r'P$_{marRAB}$', r'P$_{sodA}$', r'P$_{tolC}$', r'P$_{AM}$',
              r'P$_{MA}$', 'Control']

    # here are the names for the corresponding stress values. Stress values are the marginal distributions from Figure 1
    folder_names = ['SAK_stress', 'SFK_stress', 'SIK_stress', 'SMK_stress', 'SSK_stress', 'STK_stress']
    stress_titles = ['Wild type', r'MarA$^+$', '1mM Sal', '3mM Sal', 'lowCipro', 'highCipro']

    meta_data = []

    for l, folder in enumerate(folder_names):
        print(folder)
        time_point = []
        names = mylistdir(folder)

        names = sorted(names, key=lambda x: int(x.split('.')[0]))


        for name in names:
            mat_contents = sio.loadmat(folder + '/' + name)

            sample_data = mat_contents['total_green']

            # >>If you want to truncate any of the data this is where you should do it
            time_point.append(sample_data)

        meta_data.append(time_point)

    total_data_green = []
    total_data_red = []

    names = mylistdir('omni_data_total')
    input = []
    output = []

    for name in names:
        print(name)
        mat_contents = sio.loadmat('omni_data_total/' + name)

        sample_red = mat_contents['total_green']

        sample_data = mat_contents['total_red']

        total_data_green.append(sample_red[0][sample_red[0] > 700])
        total_data_red.append(sample_data[0][sample_red[0] > 700])

    # >>If you want to truncate any of the data this is where you should do it



    new_folder = 'output_graphs/'

    new_folder = 'output_graphs/'

    fig = Total_graphage(meta_data, stress_titles, total_data_red, total_data_green, titles, 10, best_fit=True,
                            normalized=False)



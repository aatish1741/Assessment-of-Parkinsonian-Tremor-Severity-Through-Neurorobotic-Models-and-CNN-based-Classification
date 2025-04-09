from GenericBG import Structure
from netpyne import sim
import matplotlib.pyplot as plt
import numpy as np

class MarmosetBG( Structure ):
    def __init__(self, has_pd=True, has_dbs = True, n_channels=1, seed=2):
        super( MarmosetBG, self ).__init__(has_pd=has_pd,  dbs = has_dbs,  n_channels=n_channels, seed=seed)
        self.set_marmoset()


    def set_marmoset( self ):
        genotype = [ 0.47148627, 0.         , 1.        , 0.24748603, 1.        , 0.0930143,
                     1.        ,  0.01593558, 0.9980333 , 0.19937477, 1.        , 0.48306456,
                     0.48590738,  0.14889275 ]
        self.set_genotype( genotype )


if __name__=='__main__':
    #sim.cfg.savePickle = True
    #sim.cfg.saveJson = False
    #sim.cfg.filename = 'mySimData'
    num_pd_patients =1
    num_healthy_patients = 0
    num_pd_dbs_patients = 0
    pd_coefficients = []
    healthy_coefficients = []
    dbs_coefficients = []
    all_coefficients = []
    network = MarmosetBG(has_pd = True, has_dbs =False, n_channels=1, seed=8)
    network.simulate()
    sim.saveData(filename='PD')
    sim.analysis.plotRaster(showFig=False, saveFig=True, figName='PD.png')
    sim.analysis.plotSpikeHist(showFig=False, saveFig=True, figName='PD.png')
    network.plotElaborate()
    '''
    for i in range(num_pd_patients):
        print('PD patient')
        network = MarmosetBG(has_pd = True, has_dbs = False, n_channels=1, seed=i+1)
        f_pd,PSD_PD = network.simulate()
        #print(PDcoeff)
        #pd_coefficients.extend(PDcoeff)
    #pd_mean_patients = np.array(pd_coefficients).mean(0)
    #all_coefficients.append(pd_coefficients)
    for i in range(num_healthy_patients):
        print('healthy patient')
        network = MarmosetBG(has_pd = False,  has_dbs = False, n_channels=1, seed=i+1)
        f_h,PSD_h = network.simulate()
        #healthy_coefficients.extend(PDcoeff)
    #healthy_mean_patients = np.array(healthy_coefficients).mean(0)
    #all_coefficients.append(healthy_coefficients)
    print(all_coefficients)
    for i in range(num_pd_dbs_patients):
        print('DBS patient')
        network = MarmosetBG(has_pd = True,has_dbs = True, n_channels=1, seed=i+1)
        f_dbs,PSD_dbs= network.simulate()
        #dbs_coefficients.extend(PDcoeff)
    #all_coefficients.append(dbs_coefficients)
    fig, ax = plt.subplots()
    ax.plot(f_pd[0:500], PSD_PD[0:500], color='lightcoral',label = 'Parkinson')  # Plot the first dataset in blue
    ax.plot(f_pd[0:500], PSD_h[0:500], color='chartreuse', label='Healthy')  # Plot the second dataset in red
    ax.plot(f_pd[0:500], PSD_dbs[0:500], color='orchid', label='Parkinson + DBS')  # Plot the second dataset in red
    ax.axvspan(8, 30, facecolor='lightgreen', alpha=0.3, label="Beta band")
    ax.set_xlabel('Freq (Hz)')  # ... with axes labeled
    ax.set_ylabel('Power (Hz)')
    ax.set_title('Neuron-averaged Power Spectrum Density of spiking data')
    ax.legend()
    plt.show()
    '''
    #plt.boxplot(all_coefficients, labels=['PD patients'] + [' Healthy patients'] + [' PD + DBS patients'])
    #plt.xlabel('Patient Type')
    #plt.ylabel('beta-band Relative Power')
    #plt.title('Comparison of beta-band Relative Power between 3 states')
   # plt.show()
   # network = MarmosetBG()
   # network.simulate()
   # spikes_dict = network.extractSpikes()
   # print('SPIKES')
   # for key in spikes_dict.keys():
   #     print(key)
   #     print([s.times for s in spikes_dict[key]])
   # print('LFP DATA', sim.allSimData['LFPCells'])

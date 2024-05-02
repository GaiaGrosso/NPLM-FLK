import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--signal', type=int, help="signal", required=True)
parser.add_argument('-t', '--toys', type=int, help="toys", required=True)
args = parser.parse_args()


# problem definition
N_ref = 50000 # size sample 0
N_bkg = 10000 # nr of class 0 events in sample 1
N_sig = args.signal # nr of class 1 events in sample 1
w_ref = N_bkg*1./N_ref
# Poisson fluctuations:
# set it True if the number of events is sample 1 should vary according to poisson with expectation N_bkg+N_sig
poisson_fluctuations=False 

# hyper-parameters of the kernel methods
# -- nr of kernels (M):
M=10000
# -- percentile of the pair-wise distance distribution to use to define th ekernel width (flk_sigma)
flk_sigma_perc=90 #(you can keep it fixed for now)
# -- L2 regularization (lam) keep it small! (but be aware that execution time increases for smaller lambda)
lam =1e-6
# -- number of maximum newton iterations before end (early stopping is in place)
iterations=1000000
# -- number of toy toys to get the test distribution
Ntoys = args.toys

folder_out = './'
# define folder name
NP = '%_NR%i_NB%i_NS%i_M%i_lam%s_iter%i/'%(N_ref, N_bkg, N_sig, M, str(lam), iterations)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)
    
# read all data (write your own code for this)
# note: should be more points than N_ref+N_data so you can bootstrap from it in the loop later on
print('Load all data')
features_SIG = # ... numpy array
features_BKG = # ... numpy array

# standardize using sample 0 as a reference
print('standardize')
features_mean, features_std = np.mean(features_BKG, axis=0), np.std(features_BKG, axis=0)
print('mean: ', features_mean)
print('std: ', features_std)
features_BKG = standardize(features_BKG, features_mean, features_std)
features_SIG = standardize(features_SIG, features_mean, features_std)
# compute the kernel width accoring to a specified percentile of the pair-wise distance distribution in a fraction of sample 0
flk_sigma = candidate_sigma(features_BKG[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

print('Start running toys')
ts=np.array([])
seeds = np.arange(Ntoys)+datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
for i in range(Ntoys):
    seed = seeds[i]
    rng = np.random.default_rng(seed=seed)
    N_bkg_p=N_bkg
    N_sig_p=N_sig
    if poisson_fluctuations:
        N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
        N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
    rng.shuffle(features_BKG)
    rng.shuffle(features_SIG)
    features_s = features_SIG[:N_sig_p, :]
    features_b = features_BKG[:N_bkg_p+N_ref, :]
    features  = np.concatenate((features_s,features_b), axis=0)

    label_R = np.zeros((N_ref,))
    label_D = np.ones((N_bkg_p+N_sig_p, ))
    labels  = np.concatenate((label_D,label_R), axis=0)                                                                                      
    
    plot_reco=False
    verbose=False
    if not i%20:
        plot_reco=True
        verbose=True
        
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
    t, pred = run_toy(manifold, features, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path='./', plot=plot_reco,
                      verbose=verbose)

    ts = np.append(ts, t)

# check is other simulations are already been produced, 
# if so load them and append them to new ones
seeds_past = np.array([])
ts_past = np.array([])
if os.path.exists('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    ts_past = np.array(f.get(str(flk_sigma) ) )
    f.close()
ts = np.append(ts_past, ts)
seeds = np.append(seeds_past, seeds)

# save simluations and seeds
f = h5py.File('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma), 'w')
f.create_dataset(str(flk_sigma), data=ts, compression='gzip')
f.create_dataset('seed_toy', data=seeds, compression='gzip')
f.close()
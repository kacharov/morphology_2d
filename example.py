#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:40:52 2023

@author: kacharov
"""

import numpy as np

import emcee
from corner import corner

import morphology_2d as m2d

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

##############################################################################
# Mock data generation section
##############################################################################

# Generate a mock cluster to analyse
x, y = m2d.gen_mock_plummer_cluster(10**4, rs=3.5, q=0.9, theta=0.5, background=1,
                                    x0=-0.2, y0=0.3, plot=True)


##############################################################################
# MCMC data fitting section
##############################################################################

# Begin MCMC
nfree = 6       # number of free parameters
nwalkers = 40   # number of walkers in the MCMC chain
nsteps = 600    # number of steps in the MCMC chain

# fill in initial position for the MCMC with each free parameter on a separate row
# in the same order that is read by the likelihood function.
# The ndarray samples nwalkers initial values given a mean and a standard deviation
# for each free parameter.
pos_ini = np.array([np.random.normal(1.0, 0.02, nwalkers),
                    np.random.normal(0.8, 0.02, nwalkers),
                    np.random.normal(0.78, 0.02, nwalkers),
                    np.random.normal(0.0, 0.02, nwalkers),
                    np.random.normal(0.0, 0.02, nwalkers),
                    np.random.normal(0.01, 0.002, nwalkers)])

pos_ini = np.transpose(pos_ini)

# run the MCMC chain
sampler = emcee.EnsembleSampler(nwalkers, nfree, m2d.lnprob_small_fov,
                                args=[x, y])
sampler.run_mcmc(pos_ini, nsteps, progress=True)


##############################################################################
# Best fit MCMC analysis section
##############################################################################

# Analyse MCMC
chain = sampler.chain
lnprob = np.transpose(sampler.get_log_prob())

'''
#check if there are stuck walkers and remove them
bad_walkers = np.array([],dtype=int)
for i in range(chain.shape[2]):
    for j in range(chain.shape[0]):
        if np.all(chain[j,:,i] == chain[j,0,i]):
            if j not in bad_walkers:
                bad_walkers = np.append(bad_walkers,j)
chain = np.delete(chain, bad_walkers, axis=0)
'''

labels = ['rs', 'q', 'theta', 'x0', 'y0', 'bg']

fig, axs = plt.subplots(chain.shape[2], 1, sharex=True, figsize=(9, 15))
for j in range(chain.shape[2]):
    for i in range(chain.shape[0]):
        axs[j].plot(np.arange(chain.shape[1]),
                    chain[i, :, j], c='grey', alpha=0.2)
    axs[j].set_ylabel(labels[j])
axs[j].set_xlabel('Step')
plt.tight_layout()

lnprob_fig = plt.figure(figsize=(9, 2))
for i in range(chain.shape[0]):
    plt.plot(np.arange(lnprob.shape[1]), lnprob[i, :], c='grey', alpha=0.2)
plt.xlabel('Step')
plt.ylabel('ln P')

burnin_steps = int(0.75*chain.shape[1])
samples = chain[:, burnin_steps:, :].reshape((-1, chain.shape[2]))

corner1 = corner(samples, labels=labels, show_titles=1,
                 quantiles=[0.25, 0.5, 0.75])

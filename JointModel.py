# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:08:34 2016

Initial implementation of a joint model for longitudinal and binary data.
Use a joint model to model lab data and use this to predict if encounter will
be billed as sepsis.

Model:

y_{ik} ~ Fixed effect + Random effect + t-dist error
outcome model ~ baseline covariate effect + long. random effect + med effect + random effect

Fit model with SG-HMC.  Need to integrate out local latents, so use HMC to draw
samples and MC approx the expectation.  Then use SG-HMC to draw samples from 
posterior for globals.

Use autograd to get relevant gradients. 

@author: jfutoma
"""

import pdb
import sys
from time import time
import cPickle as pickle
import glob
from copy import deepcopy

import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import value_and_grad, grad
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd.util import flatten_func, check_grads

from util import load_labs_meds, load_encounter_info

def abs_gamma_logpdf(x,shape,rate):
    """ logpdf of gamma dist, but defined on entire real line 
    by taking abs to mirror at 0.  ignores norm const. """
    return (shape-1)*np.log(np.abs(x)) - rate*np.abs(x)

def sigmoid(x):
    """ may want to experiment with other link functions for outcome model """
    return 1.0/(1.0+np.exp(-x))

class Subject():
    """ helper object to store all relevant data, precomputed values, etc
        associated with a single subject 
    """
    
    def __init__(self,cov,sepsis,labs,meds):
        #load in data
        self.cov = cov
        self.sepsis = sepsis
        
        self.labs = []
        self.X_labs = [] #design matrices for each lab submodel, for fixed effects
        self.U_labs = [] #design matrices for labs, random effects 

        #create indicator vector for which labs have been ordered; 
        #broken down by arterial just for pH, PCO2, Bicarbonate.
        self.lab_features = np.zeros(dim_lab_features) #overall indicator per lab; any arterial
        for k in xrange(K):
            lab = labs[k]
            if len(lab)==0:
                self.labs.append(np.array([]))
                self.X_labs.append(np.array([]))
                self.U_labs.append(np.array([]))
            else:
                self.lab_features[k] = 1.0 #at least one lab was taken
                if k in arterial_lab_ind: #if a lab with an arterial flag
                    if np.sum(lab[:,2])>0:
                        self.lab_features[-1] = 1.0 #an arterial lab was taken
                #remove NAs for lactate (shouldn't be any others)       
                ind = np.logical_not(np.isnan(lab[:,1]))
                lab = lab[ind,:]
                #sort rows of lab by time
                labtimes = lab[:,0]
                ind = np.argsort(labtimes)
                lab = lab[ind,:]; labtimes = lab[:,0]
                self.labs.append(lab[:,1])
                n = np.shape(lab)[0]
                #create and save design matrix X
                #just a fixed effect for intercept and slope
                #but some labs have an arterial/venous distinction indicator
                if k in arterial_lab_ind: #if a lab with an arterial flag
                    X = np.zeros((n,d*(q+1)))
                    X[:,:q] = cov
                    X[:,q] = lab[:,2] #arterial indicator (intercept)
                    X[:,(q+1):-1] = np.outer(labtimes,cov) #cov * time (slope)
                    X[:,-1] = labtimes*lab[:,2] #arterial indicator * time (slope)
                else:
                    X = np.zeros((n,d*q))
                    X[:,:q] = cov #intercept
                    X[:,q:] = np.outer(labtimes,cov) #slope
                self.X_labs.append(X)
                U = np.ones((n,2))
                U[:,1] = labtimes
                self.U_labs.append(U)

        self.num_labs = np.array([np.shape(x)[0] for x in self.labs])
        self.nz_lab_ind = np.where(self.num_labs)[0]

        #create feature vector for meds ordered
        #note that, for now, this is just indicators and so we actually 
        #never need to save these once we have the feature vectors
        self.med_features = np.zeros(dim_med_features) #3 = how many features per med to build
        for m in xrange(M):
            num = np.shape(meds[m])[0]
            if num >= 1:
                self.med_features[3*m] = 1.0 #indicator for first time
            if num >= 2:
                self.med_features[3*m+1] = 1.0 #indicator for second time
            if num >= 3:
                self.med_features[3*m+2] = num-2 #number of additional times administered
               
        #init params
        self.params = {'b': rs.normal(0,.01,dim_b), #lab random effects (ints/slopes)
                       'c': rs.normal(0,.01), #outcome random effect (int)
                       'phi': np.ones(K)} #t-dist scale params
           
        #get flattened version of posterior, gradient functions
        #note that flattened funcs are flat only wrt first arg,
        #ie flattened_local_post is flat on params but not on globalparams
        self.flattened_local_post, self.unflatten_params, self.flattened_params = flatten_func(self.neglog_posterior_local, self.params)   
        self.flattened_global_post,_,_ = flatten_func(self.neglog_posterior_global, globalparams)   
        self.grad_logpost_local = grad(self.flattened_local_post)
        self.grad_logpost_global = grad(self.flattened_global_post)

    def neglog_posterior_global(self,globalparams,params): 
        """ wrapper, used when we need grad wrt globals. only difference is arg order.
        """       
        return self.neglog_posterior_local(params,globalparams)

    def neglog_posterior_local(self,params,globalparams): 
        """ conditional posterior of local variables, z|x,\theta
        """       
        val = self.log_local_prior(params,globalparams)
        val = val + self.log_local_lik(params,globalparams)
        return -val
        
    def log_local_prior(self,params,globalparams):
        """ returns log-prior for local latents for one subject """
        Sigma_b = np.dot(globalparams['L_b'],globalparams['L_b'].T)
        tau_c = globalparams['tau_c']
        val = mvn.logpdf(params['b'],np.zeros(dim_b),Sigma_b) + norm.logpdf(params['c'],0,np.power(tau_c,-0.5))
        val = val + np.sum(abs_gamma_logpdf(params['phi'],nu_phi/2.0,nu_phi/2.0))
        return val
    
    def log_local_lik(self,params,globalparams):
        """ returns local log-likelihood for one subject:
        sum of logliks for the labs measures in this subject,
        and the loglik for the outcome model
        """
        val = 0.0
        #lab ll's 
        for k in self.nz_lab_ind: 
            mn = (np.dot(self.X_labs[k],globalparams['Lambda_'+str(k)]) +
                  np.dot(self.U_labs[k],params['b'][k*d:(k+1)*d]))
            cov = 1.0/(np.abs(globalparams['tau'][k])*np.abs(params['phi'][k]))*np.eye(self.num_labs[k])
            val = val + mvn.logpdf(self.labs[k],mn,cov)
        #outcome ll
        linpred = (np.dot(self.lab_features,globalparams['labfeat_coef']) + 
                   np.dot(params['b'],globalparams['b_coef']) + 
                   np.dot(self.med_features,globalparams['medfeat_coef']) +
                   np.dot(self.cov,globalparams['cov_coef']) +
                   params['c'])  
        prob = sigmoid(linpred)
        val = val + self.sepsis*np.log(prob+1e-12) + (1-self.sepsis)*np.log(1-prob+1e-12)
        return val
               
    def sample_params(self,flat_globalparams):
        """
        run HMC to sample from conditional of z | x, \theta.
        use these samples to get grad of marginalized ll, x | \theta
        """
        param_smps = []
        x = deepcopy(self.flattened_params)
        globalparams = unflatten_globalparams(flat_globalparams)
        
        for i in xrange(local_iter):
            #sample via HMC
            x = HMC(self.flattened_local_post,self.grad_logpost_local,x,globalparams)
            if i >= local_burn:
                param_smps.append(self.unflatten_params(x))
                
        #at end, assign current value to be last sample        
        self.flattened_params = x
        self.params = self.unflatten_params(x)
        return param_smps
    
    def grad_marginal_loglik(self,flat_globalparams):
        """
        get grad_\theta( log(p(x|\theta)) ) by using a Monte Carlo approximation of:
            grad_\theta( log(p(x|\theta)) ) = E_{z|x,\theta}[grad_\theta log(p(x,z|\theta))]
        by drawing samples from the conditional of z|x,\theta (using HMC)
        """
        localparam_smps = self.sample_params(flat_globalparams)
        grad_out = np.zeros(n_globals)
        for x in localparam_smps:
            grad_out += 1.0/local_smps*self.grad_logpost_global(flat_globalparams,x)
        return grad_out

def HMC(U,grad_U,current_q,globalparams,eps=2.5e-3,L=10):
    """
    Leapfrog method for HMC. 
    To be used in local step to sample from the conditional z_i | x_i, \theta
    These samples then used to do an MC integration in order to
    approx grad of log p(x|\theta).
    
    Inputs:
        U: objective function (-log posterior) that takes in flattened params
        grad_U: gradient of U
        current_q: current flattened params vec
        globalparams: global arg needed for U and grad_U
        eps: step size
        L: number of steps
    """
    q = deepcopy(current_q) #current local params
    p = rs.normal(0,.01,len(q))
    current_p = p

    #cache = np.ones(len(q)) #RMSprop history
    
    #half-step for momentum at beginning
    g = grad_U(q,globalparams)
    #cache = 0.9*cache + 0.1*np.power(g,2.0) 
    #eta = eps/np.sqrt(cache+1e-8)   
    eta = eps
    p -= eta*g/2.0
    
    for i in xrange(L):
        q += eta*p
        #full step for momentum except at end
        if i != L-1:
            g = grad_U(q,globalparams)
            #cache = 0.9*cache + 0.1*np.power(g,2.0) 
            #eta = eps/np.sqrt(cache+1e-8)    
            p -= eta*g
            
    #half-step for momentum at end
    g = grad_U(q,globalparams)
    #cache = 0.9*cache + 0.1*np.power(g,2.0) 
    #eta = eps/np.sqrt(cache+1e-8)    
    p -= eta*g/2.0
    
    #negate momentum to keep proposal symmetric
    p = -p
    
    #Evaluate U,K at start, end
    current_U = U(current_q,globalparams)
    current_K = np.sum(current_p**2.0)/2.0
    proposed_U = U(q,globalparams)
    proposed_K = np.sum(p**2.0)/2.0
    
    #MH step to accept reject proposal
    if rs.uniform() < np.exp(current_U-proposed_U+current_K-proposed_K):
        return q
    else:
        return current_q    

def neglog_global_prior(globalparams):
    """ log-prior for all global latents """
    Sigma_b = np.dot(globalparams['L_b'],globalparams['L_b'].T)
    val = (-0.5*(1+dim_b+nu_Sig_b)*np.linalg.slogdet(Sigma_b)[1] 
           -0.5*np.trace(np.linalg.solve(Sigma_b,Phi_Sig_b))) 
    val = val + np.sum(abs_gamma_logpdf(globalparams['tau'],a_tau,b_tau)) 
    val = val + abs_gamma_logpdf(globalparams['tau_c'],a_tau_c,b_tau_c) 
    val = val - 0.5/coef_sig2*np.sum(np.power(globalparams['labfeat_coef'],2.0))   
    val = val - 0.5/coef_sig2*np.sum(np.power(globalparams['b_coef'],2.0))   
    val = val - 0.5/coef_sig2*np.sum(np.power(globalparams['medfeat_coef'],2.0))
    val = val - 0.5/coef_sig2*np.sum(np.power(globalparams['cov_coef'],2.0))
    for k in xrange(K):
        val = val - 0.5/coef_sig2*np.sum(np.power(globalparams['Lambda_'+str(k)],2.0))   
    return val

def init_global_params():
    """ returns dict with global params randomly initialized """
    globalparams = {'L_b': 0.1*np.eye(dim_b), #cholesky of Sigma_b, np.dot(L_b,L_b.T)=Sigma_b
                    'labfeat_coef': rs.normal(0,.001,dim_lab_features),
                    'b_coef': rs.normal(0,.001,dim_b),
                    'medfeat_coef': rs.normal(0,.001,dim_med_features),
                    'cov_coef': rs.normal(0,.001,q),
                    'tau_c': 100.0,
                    'tau': np.ones(K)}
    for k in xrange(K):
        if k in arterial_lab_ind: 
            globalparams['Lambda_'+str(k)] = rs.normal(0,.001,d*(q+1))
        else:
            globalparams['Lambda_'+str(k)] = rs.normal(0,.001,d*q)
    return globalparams

def make_subjects(train_or_test_ids):
    """
    make a train or test set from set of encounter IDs
    """
    data = {}    
    for i in xrange(len(train_or_test_ids)):
        ID = train_or_test_ids[i]
        labs = []
        for k in xrange(K):
            if ID in labs_data[k]:
                labs.append(labs_data[k][ID])
            else:
                labs.append(np.array([]))    
        meds = []
        for m in xrange(M):
            if ID in meds_data[m]:
                meds.append(meds_data[m][ID])
            else:
                meds.append(np.array([]))            
        data[ID] = Subject(cov_data[ID],sepsis_outcomes[ID],labs,meds)    
    
    return data
    
if __name__ == "__main__":
        
    rs = np.random.RandomState(0) #fixed seed. always use 0 to ensure same train/test split, for paper draft.      

    #####
    ##### define relative paths 
    #####
    
    encounters_path = './cleaned_data/encounters_info.csv'
    labs_folder = './cleaned_data/labs/'
    meds_folder = './cleaned_data/meds/'
    outpath = './test_results.p'      
    
    #get names of all labs and meds 
    lab_names = [path.split('/')[-1].split('.')[0] for path in glob.glob(labs_folder+'*')]
    med_names = [path.split('/')[-1].split('.')[0] for path in glob.glob(meds_folder+'*')]
   
    #####
    ##### load data
    #####
        
    #overall int, baseline age, transfer, not elective, male, not white, log(los)    
    cov_data, sepsis_outcomes, age_loglos_stats = load_encounter_info(encounters_path) 
    N = len(cov_data) #total number of encounters
    IDs = np.array(cov_data.keys())
    
    ### decided to make these lists for simplicity, rather than dicts keyed on names
    ### use lab_names and med_names to keep in mind indices of particular labs/meds
    ### don't forget only pH, PCO2, Bicarbonate have arterial flags
    labs_data = load_labs_meds(labs_folder) #don't forget 'mean' and 'std' are keys also
    meds_data = load_labs_meds(meds_folder)
    
    K = len(labs_data)
    M = len(meds_data) 
    arterial_lab_ind = np.array([4,18,19]) #Bicarbonate, PCO2, pH   
    
    #so, lots missing
    enc_lab_nums = zip(lab_names,[len(labs)-2 for labs in labs_data])
    enc_med_nums = zip(med_names,[len(meds) for meds in meds_data])
    
    #####
    ##### prior/model settings
    #####
               
    #dimensions
    q = len(cov_data[IDs[0]]) #number of covariates, including intercept
    d = 2 #dimension of basis used.  for linear basis [1,t], d=2. 

    dim_lab_features = K*1+1       
    dim_med_features = 3*M
    dim_b = 2*K       
       
    #priors
    nu_phi = 3.0 #t-dist df for labs
    a_tau_c = 0.01 #Ga prior on tau_c
    b_tau_c = 0.01        
    a_tau = 1.0 #Ga prior on the tau's  
    b_tau = 1.0    
    nu_Sig_b = dim_b+1 #IW prior on Sigma_b
    Phi_Sig_b = np.eye(dim_b)
    coef_sig2 = 25.0 #indep zero mean norm priors on all coefs in lab/outcome models
       
    #####
    ##### initialize global model parameters
    #####   
    
    globalparams = init_global_params()
    
    #####
    ##### create train set
    #####
    
    trainN = 1000
    trainIDs = IDs[rs.permutation(N)[:trainN]]
    trainData = make_subjects(trainIDs)
    print("data loaded and training set created!")            
    
    #####
    ##### define functions for global steps; local steps in Subject class
    ##### 
    
    flattened_global_prior, unflatten_globalparams, flattened_globalparams = flatten_func(neglog_global_prior, globalparams)        
    grad_global_prior = grad(flattened_global_prior)  
    n_globals = len(flattened_globalparams)    
    
    #####
    ##### learning and optimization settings
    #####    
        
    burn = 0
    totIter = 1000
    thin = 1
    smps = (totIter-burn)/thin 
            
    track_vals = np.zeros((3,totIter)) #local logpriors, local logliks, global logprior       
            
    batchSize = 100   
    A = 0.1 #SG-HMC constant
    step_size = 0.1 #should this change over time?
    
    local_iter = 1000 #params for MC integration 
    local_burn = 0
    local_smps = local_iter-local_burn
    
    #####
    ##### save samples. last dim of array is always samples
    #####
        
    smp_ind = 0 #index of current sample
    saved_smps = np.zeros((n_globals,smps)) #save unflattened vec; unflatten later
    
    subsample_perm = rs.permutation(trainN)
    perm_ind = 0
    momentum = rs.normal(0,.01,n_globals)
    #####
    ##### main sampling loop
    #####    
    for iteration in xrange(totIter):
                     
        ##### 
        ##### local step:
        #####   run a few steps of HMC for each subject in minibatch
        #####   to approx the gradient of local loglik, marginalized over locals      
        #####
        iter_start = time()
        print("starting iter %d" %iteration)
                
        #####     
        ##### get minibatch
        #####     
        if (perm_ind + batchSize) > N:
            subsample_perm = rs.permutation(N)
            perm_ind = 0
        subsamp_inds = subsample_perm[perm_ind:(perm_ind+batchSize)]
        subsamp_ids = trainIDs[subsamp_inds]
        perm_ind += batchSize
                
        #init grad of log-posterior of globals     
        global_post_grad = np.zeros(n_globals)
                
        for i in xrange(batchSize):          
            dat = trainData[subsamp_ids[i]]         
            #add in grad of marginalized ll for each observation
            global_post_grad += float(trainN)/batchSize*dat.grad_marginal_loglik(flattened_globalparams) 
            track_vals[0,iteration] += float(trainN)/batchSize*dat.log_local_prior(dat.params,globalparams)
            track_vals[1,iteration] += float(trainN)/batchSize*dat.log_local_lik(dat.params,globalparams)
                                               
        #####
        ##### global step: resample global params using SG-MCMC
        #####   results of local step give a SG of marginalized data loglik
        #####   combine with gradient of log prior to get gradient of log post                                             
        #####   use this to resample new values for globals using SG-HMC update                                              
        ##### 
        
        global_post_grad -= grad_global_prior(flattened_globalparams)
        track_vals[2,iteration] = -neglog_global_prior(globalparams)      
               
        #Update
        flattened_globalparams += step_size*momentum      
        momentum += -step_size*global_post_grad - A*step_size*momentum + np.sqrt(2*A*step_size)*rs.normal(0,1,n_globals)
               
        globalparams = unflatten_globalparams(flattened_globalparams)         
               
        iter_end = time()
        print('took %.2f, global prior %.4f, local prior %.4f, local ll %.4f' 
            %(iter_end-iter_start,track_vals[2,iteration],track_vals[0,iteration],track_vals[1,iteration]) )
        
        #####
        ##### save globals if past burn-in
        #####
        if iteration >= burn & iteration % thin == (thin-1):
            saved_smps[:,smp_ind] = flattened_globalparams
            smp_ind += 1    
        
        #####
        ##### and save results out every so often
        #####
        if iteration % 1 == 0:
            print("saving!")
            dict_to_save = {'iteration': iteration,
                            'saved_smps': saved_smps,
                            'momentum': momentum,
                            'track_vals': track_vals,
                            'globalparams': globalparams}                                      
            pickle.dump(dict_to_save,open(outpath,'wb'))   

            
            
            
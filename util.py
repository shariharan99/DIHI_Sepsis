# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:08:53 2016

Some utility functions to read in the data and perform other misc tasks 
that don't need to be in the main modeling script.

@author: jfutoma
"""

import autograd.numpy as np
import glob

def rowvec(x):
    """
    Casts an array as a row vector.
    """
    y = np.ravel(x)
    return y[None, :]

def colvec(x):
    """
    Casts an array as a column vector.
    """
    y = np.ravel(x)
    return y[:, None]
    
def load_encounter_info(path):
    """
    return baseline covariates and outcome (binary: encounter billed as sepsis)
    as a dict of arrays and dict of ints; where key is encounter ID
    
    also:
        center/scale age variable
        add a column of 1's for overall intercept
        take log of LOS (very right skewed), then center/scale
        
    covariates: overall int, baseline age, transfer, not elective, male, not white, los
    """
    dat = np.genfromtxt(path,skip_header=1,delimiter=',')
    IDs = dat[:,0]; IDs = np.array([int(ID) for ID in IDs]) 
    sepsis_codes = dat[:,1]
    cov_data = dat[:,2:] #strip off IDs, outcomes
    cov_data = np.insert(cov_data,0,np.ones(np.shape(cov_data)[0]),1) #add an initial column of 1's for intercept
    
    #center and scale the age covariate
    mean_age = np.mean(cov_data[:,1]); std_age = np.std(cov_data[:,1])
    cov_data[:,1] = (cov_data[:,1] - mean_age)/std_age      
    
    #log LOS; then center/scale    
    cov_data[:,6] = np.log(cov_data[:,6])
    mean_log_los = np.mean(cov_data[:,6]); std_log_los = np.std(cov_data[:,6])
    cov_data[:,6] = (cov_data[:,6] - mean_log_los)/std_log_los       
    
    cov_dict = {}
    sepsis_dict = {}
    for i in xrange(len(IDs)):
        ID = IDs[i]
        cov_dict[ID] = cov_data[i,:]
        sepsis_dict[ID] = sepsis_codes[i]
        
    #will want the means and variances if we need to un-standardize later    
    return cov_dict, sepsis_dict, (mean_age,std_age,mean_log_los,std_log_los)
    
def load_labs_meds(path):
    """
    takes in path of folder containing all lab/med data files.
    
    note that we build out the indicators for which labs have been ordered 
    for an encounter when we create Subject instances in modeling file.    
    
    returns labs/meds as a list of dicts. 
    each dict is data for each lab/med, keyed by encounter id.
        times and values (and arterial flag) for flags
        just times for meds (may want to change later and account for med subtypes)
    """
    files = glob.glob(path+'*') #list with all lab files
    K = len(files) #number of labs
    dat_dicts = [] 
    
    for k in xrange(K):
        path = files[k]
        dat_dicts.append(load_lab_med(path))       
        
    return dat_dicts

def load_lab_med(path):
    """ helper; load a single lab/med (generic, assuming same format)
    NOTE: this implicitly assumes that the data files are already 
          sorted by encounter so that a single quick pass over the 
          file suffices. we don't assume for a given encounter that 
          labs are chronological; will sort when Subjects are created.
    """
    dat = np.genfromtxt(path,skip_header=1,delimiter=',')   
    IDs = dat[:,0]; IDs = np.array([int(ID) for ID in IDs]) 
    dat_dict = {}
    
    if np.shape(dat)[1]>2: #if lab, first convert to log scale, then standardize
        dat[:,2] = np.log(np.maximum(dat[:,2],.01))
        mean = np.nanmean(dat[:,2])
        std = np.nanstd(dat[:,2])
        dat[:,2] = (dat[:,2]-mean)/std
        #add to the dict that we return
        dat_dict['mean'] = mean
        dat_dict['std'] = std
        
    start = 0 #start of data for this person
    cur_id = IDs[start] #id of current person
    
    for i in xrange(len(IDs)):
        if IDs[i] != cur_id:
            dat_dict[cur_id] = dat[start:i,1:] 
            cur_id = IDs[i]
            start = i
    dat_dict[cur_id] = dat[start:,1:] #edge case at end             
            
    return dat_dict


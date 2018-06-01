# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:26:07 2018

@author: kisha_000
"""


import numpy as np
import math


def cluster_indices(cluster_assignments,idx):
    n = cluster_assignments.max()
    clusters = []
    for cluster_number in range(0, n + 1):
        aux = np.where(cluster_assignments == cluster_number)[0].tolist()
        cluster = list(idx[i] for i in aux )
        clusters.append(cluster)
    return clusters

def cluster_external_index(partition_a, partition_b):
    #size of contigency table
    R = len(partition_a)
    C = len(partition_b)
    #contigency table
    ct = np.zeros((R+1,C+1))
    #fill the contigency table
    for i in range(0,R+1):
        for j in range(0,C):
            if(i in range(0,R)):  
                n_common_elements = len(set(partition_a[i]).intersection(partition_b[j]))
                ct[i][j] = n_common_elements
            else:
                ct[i][j] = ct[:,j].sum()
                      
        ct[i][j+1] = ct[i].sum()  
    
    N = ct[R][C]
    #condensed information of ct into a mismatch matrix (pairwise agreement)
    sum_all_squared = np.sum(ct[0:R][:,range(0,C)]**2)   
    sum_R_squared = np.sum(ct[0:R,C]**2)
    sum_R = np.sum(ct[0:R,C])
    sum_C_squared = np.sum(ct[R,0:C]**2)
    sum_C = np.sum(ct[R,0:C])
    #computing the number of pairs that are in the same cluster both in partition A and partition B
    a = 0
    for i in range(0,R):
        for j in range(0,C):
            a = a + ct[i][j]*(ct[i][j]-1)
    a = a/2
    #computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = (sum_R_squared- sum_all_squared)/2
    #computing the number of pair in different cluster in partition A but in the same cluster in partition B
    c = (sum_C_squared - sum_all_squared)/2
    #computing the number of pairs in different cluster both in partition A and partition B
    d = (N**2 + sum_all_squared - (sum_R_squared + sum_C_squared))/2
    
    #Rand Index
    rand_index = (a+d)/(a+b+c+d)

    #Adjusted Rand Index
    nc = ((sum_R_squared - sum_R)*(sum_C_squared -sum_C))/(2*N*(N-1))
    nd = (sum_R_squared - sum_R + sum_C_squared - sum_C)/4
    if(nd==nc):
        adjusted_rand_index = 0
    else:      
        adjusted_rand_index = (a-nc)/(nd - nc)
   
    #Fowlks and Mallows
    if((a+b)==0 or (a+c)==0):
        FM = 0
    else:     
        FM = a/math.sqrt((a+b)*(a+c))
    
    #Jaccard
    if(a+b+c == 0):
        jaccard = 1
    else:
        jaccard = a/(a+b+c)
        
    #Adjusted Wallace
    if((a+b)==0):
        wallace = 0
    else:
        wallace = a/(a+b)
    SID_B = 1-((sum_C_squared-sum_C)/(N*(N-1)))
    if((SID_B)==0):
        adjusted_wallace = 0
    else:
        adjusted_wallace = (wallace-(1-SID_B))/(1-(1-SID_B))

    return [rand_index, adjusted_rand_index, FM, jaccard, adjusted_wallace]


def adjusted_wallace(partition_a,partition_b):
        #size of contigency table
    R = len(partition_a)
    C = len(partition_b)
    #contigency table
    ct = np.zeros((R+1,C+1))
    #fill the contigency table
    for i in range(0,R+1):
        for j in range(0,C):
            if(i in range(0,R)):  
                n_common_elements = len(set(partition_a[i]).intersection(partition_b[j]))
                ct[i][j] = n_common_elements
            else:
                ct[i][j] = ct[:,j].sum()
                      
        ct[i][j+1] = ct[i].sum()  
    
    N = ct[R][C]
    #condensed information of ct into a mismatch matrix (pairwise agreement)
    sum_all_squared = np.sum(ct[0:R][:,range(0,C)]**2)   
    sum_R_squared = np.sum(ct[0:R,C]**2)
    sum_C_squared = np.sum(ct[R,0:C]**2)
    sum_C = np.sum(ct[R,0:C])
    #computing the number of pairs that are in the same cluster both in partition A and partition B
    a = 0
    for i in range(0,R):
        for j in range(0,C):
            a = a + ct[i][j]*(ct[i][j]-1)
    a = a/2
    #computing the number of pair in the same cluster in partition A but in different cluster in partition B
    b = (sum_R_squared- sum_all_squared)/2    
    #Adjusted Wallace
    SID_B = 1-((sum_C_squared-sum_C)/(N*(N-1)))
    wallace = a/(a+b)
    adjusted_wallace = (wallace-(1-SID_B))/(1-(1-SID_B))
    
    return adjusted_wallace

def cluster_validation_indexes(cluster_a,cluster_b):
    #jaccard index
    num_jaccard = len(set(cluster_a).intersection(cluster_b))
    den_jaccard = len(set(cluster_a).union(cluster_b))
    jaccard = num_jaccard/den_jaccard
    
    #the asymmetric measure gama - rate of recovery
    gama = num_jaccard/len(cluster_a)
    
    #the symmetric Dice coefficient
    dice = num_jaccard/(len(cluster_a)+len(cluster_b))
    
    return [jaccard, gama, dice]



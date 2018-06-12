# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:23:23 2018

@author: kisha_000
"""

# Implementation of AliClu

from matplotlib.backends.backend_pdf import PdfPages
from sequence_alignment import main_algorithm
from clustering import convert_to_distance_matrix, hierarchical_clustering
from hierarchical_validation import validation, final_decision
from cluster_stability import cluster_validation
from clustering_scores import cluster_indices
from scipy.cluster.hierarchy import cut_tree
from print_results import print_clusters_csv
import pandas as pd
import itertools
import numpy as np
import string
import argparse


#np.random.seed(123)
np.seterr(all='raise')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='AliClu')
    #positional arguments - filename, maximum number of clusters, automatic or not 
    parser.add_argument("filename",help='Input CSV file with temporal sequences for each patient')
    parser.add_argument("maxClusters",type=int,help='Maximum number of clusters to analyse on validation step')
    parser.add_argument('automatic',type=int,help='1 to run AliClu automatically, 0 otherwise')
    #optional arguments - gap penalty, temporal penalty constant, number of 
    #bootstrap samples, distance metric for hierarchical clustering
    parser.add_argument('-g','--gap',metavar='',help='Gap penalty')
    parser.add_argument('-tp','--temporalPenaltyConstant',type=float,metavar='',help='Temporal penalty constant')
    parser.add_argument('-M','--bootstrapSamples',type=int,metavar='',help='Number of bootstrap samples')
    parser.add_argument('-d','--distanceMetric',metavar='',help='Distance metric for agglomerative clustering')
    args = parser.parse_args()
    
    ###############################################################################
    #               PARAMETERS
    ###############################################################################
    #pre-defined scoring system for TNW Algorithm
    match=1.
    mismatch=-1.1
    #initialize pre-defined scoring dictionary
    s = {'11': match}
    #get all combinations of letters of alphabet
    alphabet = string.ascii_uppercase
    comb = list(itertools.product(alphabet,repeat = 2))
    #construct the pre-defined scoring system
    for pairs in comb:
        if(pairs[0]==pairs[1]):
            s[pairs[0]+pairs[1]] = match
        else:
            s[pairs[0]+pairs[1]] = mismatch
    
    #gap penalty for TNW Algorithm
    if(args.gap):
        gap_string_values = args.gap.split('[')[1].split(']')[0].split(' ')
        if(len(gap_string_values)==1):
            gap_values = [float(gap_string_values[0])]
        elif(len(gap_string_values)==3):
            min_gap = float(gap_string_values[0])    
            max_gap = float(gap_string_values[1])
            step_gap = float(gap_string_values[2])
            #gap_values = np.arange(min_gap,max_gap+step_gap,step_gap)
            num = (max_gap-min_gap)/(step_gap) + 1
            gap_values = np.linspace(min_gap,max_gap,num)
        else:
            parser.error("Gap values not in the correct format")
    else:
        gap_values = np.linspace(-1,1,21)
    #print(gap_values)
    
    #Temporal penalty for temporal penalty function of TNW Algorithm
    if(args.temporalPenaltyConstant):
        T = args.temporalPenaltyConstant
    else:
        T = 0.25
    #print(T)
    
    #number of bootstrap samples M for validation step
    if(args.bootstrapSamples):
        M = args.bootstrapSamples
    else:
        M = 250
    #print(M)
    
    #maximum number of clusters to analyse on validation step
    max_K = args.maxClusters
    #print(max_K)
    
    #distance metric used in hierarchical clustering
    if(args.distanceMetric):
        allowed_methods = ['single','complete','average','centroid','ward']
        if(args.distanceMetric in allowed_methods):
            method = args.distanceMetric
        else:
            parser.error("Not one of the 5 distance metrics used in AliClu")
    else:
        method = 'ward'
    #print(method)
    
    ###############################################################################
    #          READ TEMPORAL SEQUENCES
    ###############################################################################
    df_encoded = pd.read_csv(args.filename,sep=',')
    ################################################################################
    ##            SEQUENCE ALIGNMENT, HIERARCHICAL CLUSTERING & VALIDATION
    ################################################################################
    concat_for_final_decision = []
    #pdf file to store dendrogram, table of averages and graph of standard deviation
    #when using AliClu in a non automated way
    if(args.automatic == 0):
        pp = PdfPages('semi_automatic_analysis.pdf')
    else:
        pp = 0
    for gap in gap_values:
        
        print('Analysing with gap %.2f...'  %gap)
        
        #pairwise sequence alignment results
        results = main_algorithm(df_encoded,gap,T,s,0)
        
        #reset indexes
        df_encoded = df_encoded.reset_index()
        
        #convert similarity matrix into distance matrix
        results['score'] = convert_to_distance_matrix(results['score'])
        
        #exception when all the scores are the same, in this case we continue with the next value of gap
        if((results['score']== 0).all()):
            #print('entrei')
            continue
        else:
            #hierarchical clustering
            Z = hierarchical_clustering(results['score'],method,gap,T,args.automatic,pp)

            #validation
            chosen = validation(M,df_encoded,results,Z,method,max_K+1,args.automatic,pp,gap,T)
            chosen_k = chosen[2]
            df_avgs = chosen[0]
            df_stds = chosen[1]
            
            chosen_results = df_avgs.loc[chosen_k]
            chosen_results['gap'] = gap
            concat_for_final_decision.append(chosen_results)
    
    ############################################################################
    #       RESULTS
    ############################################################################
    if(args.automatic==1):
        df_final_decision = pd.concat(concat_for_final_decision,axis=1).T
        final_k_results = final_decision(df_final_decision)
        print('Chosen gap penalty: ',final_k_results['gap'])
        print('\n')
        print('Chosen number of clusters: ',final_k_results['k'])
        print('\n')
        final_gap = final_k_results['gap']
        k = int(final_k_results['k'])
    elif(args.automatic==0):
        #close pdf  
        pp.close()
        #USER INPUTS THE FINAL NUMBER OF CLUSTERS
        k = input("Please enter the final number of clusters: ")
        k = float(k)
        while(k>max_K or k<2):
            k = input("Please enter one of the analysed number of clusters: ")
            k = float(k)
        print("Final number of clusters: " + str(k))
        k=int(k)
        #USER INPUTS THE CHOSEN GAP PENALTY
        if(len(gap_values)>1):
            final_gap = input("Please enter the chosen gap penalty: ")
            final_gap = float(final_gap)
            while(final_gap not in gap_values):    
                final_gap = input("Please enter one of the analysed gap penalties: ")
                final_gap = float(final_gap)
            print("Chosen gap penalty: " + str(final_gap))
        else:
            final_gap = gap_values[0]
    
    #perform alignment with the chosen gap
    results = main_algorithm(df_encoded,final_gap,T,s,0)
    
    #convert similarity matrix into distance matrix
    results['score'] = convert_to_distance_matrix(results['score'])
    
    #hierarchical clustering
    Z = hierarchical_clustering(results['score'],method,gap,T,1,0)
    
    #reset indexes
    df_encoded = df_encoded.reset_index()
    
    #PRINT CLUSTERS
    #cut the final dendrogram
    c_assignments_found = cut_tree(Z,k)
    #obtain the final partitions
    partition_found = cluster_indices(c_assignments_found,df_encoded.index.tolist())
    partition_found.sort(key=len)
    directory = method + '_gap_' + str(final_gap) + '_Tp_' + str(T) + '_clusters_' + str(k) + '/'
    #print the clusters on a directory in separated csv files
    print_clusters_csv(k,partition_found,df_encoded,directory)

    #CLUSTER VALIDATION AFTER FOUNDING K
    cluster_validation(M,method,k,partition_found,df_encoded,results,final_gap,T)
   

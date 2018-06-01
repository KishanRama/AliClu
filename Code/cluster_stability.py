# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:11:38 2018

@author: kisha_000
"""

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from fastcluster import linkage
from scipy.cluster.hierarchy import cut_tree
from clustering_scores import cluster_indices,cluster_validation_indexes
import numpy as np
import pandas as pd
from statistics import mean, stdev, median
#from tabulate import tabulate
import itertools


    ##############################################################################
    #CLUSTER VALIDATION (STABILITY)
    ##############################################################################
def cluster_validation(M,method,k,partition_found,df_encoded,results,gap,Tp):
    
    #write cluster stability analysis on a pdf page
    pp = PdfPages('cluster_stability_analysis.pdf')
    
    #dictionary to store all computed indexes for each cluster
    dicio_cluster_validation = {k:{} for k in range(1,k+1)}
    for k in range(1,k+1):
        dicio_cluster_validation[k]['jaccard'] = []
        dicio_cluster_validation[k]['dice'] = []
        dicio_cluster_validation[k]['asymmetric'] = []
        

    #assess cluster stability for K=k that was the number of clusters chosen
    for i in range(M):
        # sampling rows of the original data
        idx = np.random.choice(len(df_encoded), int((3/4)*len(df_encoded)), replace = False)
        idx = np.sort(idx)
        #get all the possible combinations between the sampled patients
        patient_comb_bootstrap = list(itertools.combinations(df_encoded.loc[idx,'id_patient'],2))
        patient_comb_bootstrap = pd.DataFrame(patient_comb_bootstrap,columns = ['patient1','patient2'])
        #extract the scores regarding the previous sampled combinations to be used in hierarchical clustering
        results_bootstrap = pd.merge(results, patient_comb_bootstrap, how='inner', on=['patient1','patient2'])
        # Hierarchical Clustering of the bootstrap sample
        Z_bootstrap = linkage(results_bootstrap['score'],method)
        
        c_assignments_bootstrap = cut_tree(Z_bootstrap,k)
        partition_bootstrap = cluster_indices(c_assignments_bootstrap,idx)
        
        for k_i in range(1,k+1):
            aux_jaccard = []
            aux_dice = []
            aux_asymmetric = []
            for i in range(1,k+1):
                aux = cluster_validation_indexes(partition_found[k_i-1],partition_bootstrap[i-1])
                aux_jaccard.append(aux[0])
                aux_dice.append(aux[2])
                aux_asymmetric.append(aux[1])
            
            dicio_cluster_validation[k_i]['jaccard'].append(max(aux_jaccard))
            dicio_cluster_validation[k_i]['dice'].append(max(aux_dice))
            dicio_cluster_validation[k_i]['asymmetric'].append(max(aux_asymmetric))
            
    #obtain the average cluster external indexes for each number of clusters
    jaccard_cluster_median = []
    dice_median = []
    asymmetric_median = []
    jaccard_cluster_avg = []
    dice_avg = []
    asymmetric_avg = []
    jaccard_cluster_std = []
    dice_std = []
    asymmetric_std = []
    table = []
    
    for k in range(1,k+1):
        jaccard_cluster_median.append(round(median(dicio_cluster_validation[k]['jaccard']),3))
        dice_median.append(round(median(dicio_cluster_validation[k]['dice']),3))
        asymmetric_median.append(round(median(dicio_cluster_validation[k]['asymmetric']),3))
        jaccard_cluster_avg.append(round(mean(dicio_cluster_validation[k]['jaccard']),3))
        dice_avg.append(round(mean(dicio_cluster_validation[k]['dice']),3))
        asymmetric_avg.append(round(mean(dicio_cluster_validation[k]['asymmetric']),3))
        jaccard_cluster_std.append(round(stdev(dicio_cluster_validation[k]['jaccard']),3))
        dice_std.append(round(stdev(dicio_cluster_validation[k]['dice']),3))
        asymmetric_std.append(round(stdev(dicio_cluster_validation[k]['asymmetric']),3))
    
        table.append([str(k) + ' (' + str(len(partition_found[k-1])) + ')',
                      jaccard_cluster_median[k-1], dice_median[k-1], asymmetric_median[k-1],
                      jaccard_cluster_avg[k-1], dice_avg[k-1], asymmetric_avg[k-1], 
                      jaccard_cluster_std[k-1], dice_std[k-1], asymmetric_std[k-1]])
  
    headers = ['Cluster Number', 'J_median','D_median','A_median','J_avg','D_avg','A_avg','J_std','D_std','A_std']
    #print(tabulate(table,headers))
    
    fig = plt.figure(figsize=(10,4))
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis('tight')
    ax.axis('off')
    plt.title('Cluster stability analysis \n gap: %.2f, Tp: %.2f, %s link' %(gap,Tp,method))
    the_table = plt.table(cellText=table, colLabels=headers, loc='center',cellLoc='center')
    the_table.set_fontsize(15)
    the_table.scale(1.1, 1.1)
    pp.savefig(fig)
    pp.close()
    
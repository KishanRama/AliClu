# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:35:56 2018

@author: kisha_000
"""

from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram,cophenet
from matplotlib import pyplot as plt


#Function that receives a similarity matrix in a list form and converts it to a 
#matrix. Negate all entries and add an offset to make all values positive.
def convert_to_distance_matrix(similarity_matrix):
    
    distance_matrix = -similarity_matrix
    distance_matrix = distance_matrix + abs(distance_matrix[distance_matrix.idxmin()])
    
    return distance_matrix

#Function that performs agglomerative clustering. It receives as input a distance matrix
#and a distance metric used to measure distance between clusters.
# It outputs a dendrogram and the cophenetic correlation coefficient.
def hierarchical_clustering(distance_matrix,method,gap,Tp,automatic,pp):
    
    
    #agglomerative clustering
    Z = linkage(distance_matrix, method)
    
    if(automatic == 0 or automatic == 1):
        #dendrogram plot
        fig = plt.figure(figsize=(40, 20))
        plt.title('Hierarchical Clustering Dendrogram - gap: %.2f, Tp: %.2f, %s link'  %(gap,Tp,method),fontsize=30)
        plt.xlabel('patient index',labelpad=20,fontsize=30)    
        plt.ylabel('distance',labelpad=10,fontsize=30)    
        plt.xticks(size = 40)
        plt.yticks(size = 40)
        dendrogram(
                
                Z,
                #truncate_mode = 'lastp',
                #p=6,
                leaf_rotation=90.,  # rotates the x axis labels
                leaf_font_size=15.,  # font size for the x axis labels
                )
        #plt.show()
        
        # Cophenetic Correlation Coefficient
        c, coph_dists = cophenet(Z, distance_matrix)
        txt = 'Cophenetic Correlation Coefficient: ' + str(c)
        fig.text(0.1,0.01,txt,fontsize=30)
        
        #fig.savefig('dendrogram.png', format='png')
    
        #c = canvas.Canvas("hello.pdf")
        #c.drawString(100,750,"Welcome to Reportlab!")
        #c.drawImage()
        #c.save()
        
        pp.savefig(fig)
        

    
    return Z
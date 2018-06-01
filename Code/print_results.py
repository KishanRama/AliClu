# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:46:19 2018

@author: kisha_000
"""

import os

def print_latex_code(df_clusters,filename,cluster_number):
    latex_code = '\\textbf{Cluster '+str(cluster_number) + ' - '+ str(len(df_clusters)) + ' patients} \n' +\
                 ' \n\\vspace{3mm} \n \n' +\
                '\\begin{tabular}{cc} \n\hline \n'
    latex_code = latex_code + 'id\_patient & PE Temporal sequences \\\ \n\hline \n'
    for index,row in df_clusters.iterrows():
        latex_code = latex_code + str(row['id_patient']) +'\t & \t'+ str(row['aux_encode']) + '\t \\\ \n' 
    
    latex_code = latex_code + '\end{tabular} \n \n\\vspace{5mm} \n'

    return latex_code                            
    
#print the resulting clusters and the correspondent code for latex tables
def print_clusters(k,partition_found,df_encoded,filename):
        
    text_file = open(filename, "w")
    text_file.write(filename)
    text_file.write("\n")
    for c in range(0,k):
        text_file.write("\n")
        text_file.write("Cluster %s - %s elements" % (str(c+1) , len(partition_found[c])))
        text_file.write("\n")
        text_file.write("%s" % df_encoded.loc[partition_found[c]])
        text_file.write("\n \n")
        latex_code = print_latex_code(df_encoded.loc[partition_found[c]],filename,c+1)
        text_file.write("%s" % latex_code)
        text_file.write("\n \n")
        
    text_file.close()
    
def print_clusters_csv(k,partition_found,df_encoded,directory):
    #create directory
    try:
        if not os.path.exists('./'+directory):
            os.makedirs('./'+directory)
    except OSError:
        print ('Error: Creating directory. ' +  './'+directory)
        
    for c in range(0,k):
        cluster_name = 'Cluster '+ str(c+1) + ' - ' + str(len(partition_found[c])) + ' elements.csv'
        df_encoded.loc[partition_found[c]].to_csv(directory+cluster_name,encoding='utf-8', index=False)

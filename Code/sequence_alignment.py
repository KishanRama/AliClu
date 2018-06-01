# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:35:48 2018

@author: kisha_000
"""

#Application of Temporal Needlman-Wunsch as in the 
#paper Temporal Needlman-Wunsch by Haider Syed and Amar K. Das
#The objective is to performe alignment between all pairwise elements of a dataframe, this 
#dataframe comes with an id and the temporal sequence for each element.

import pandas as pd
import numpy as np
import itertools


# Initialisation of the score matrix
def score_initialisation(rows,cols,gap):
    
            
    score=np.zeros((rows,cols),float)

    for i in range(rows):
        score[i][0] = -i*gap
    for j in range(cols):
        score[0][j] = -j*gap
             
    return score

# Initialisation of the traceback matrix
def traceback_initialisation(rows,cols):
    
            
    traceback=np.zeros((rows,cols))

    # end of path top left corner
    traceback[0][0] = -1

    #going up
    for i in range(1,rows):
        traceback[i][0] = 1 
    
    #going left
    for j in range(1,cols):
        traceback[0][j] = 2
             
    return traceback


# Initialisation of the TR matrix
def TR_initialisation(rows,cols,traceback,seq2):
    
            
    TR=np.zeros((rows,cols))

    # end of path top left corner
    TR[0][0] = 0

    #going up
    for i in range(1,rows):
        TR[i][0] = TR[i-1][0]+float(seq2[i-1][0]) 
    
    #going left
    for j in range(1,cols):
        TR[0][j] = 0
             
    return TR

# Initialisation of the TC matrix
def TC_initialisation(rows,cols,traceback,seq1):
    
            
    TC=np.zeros((rows,cols))

    # end of path top left corner
    TC[0][0] = 0

    #going up
    for i in range(1,rows):
        TC[i][0] = 0
    
    #going left
    for j in range(1,cols):
        TC[0][j] = TC[0][j-1]+float(seq1[j-1][0]) 
             
    return TC

# calculation of the scores and filling the traceback matrix
def calculate_scores(score,traceback,rows,cols,seq1,seq2,TR,TC,gap,T,s):
       
    for i in range(1,rows):
        for j in range(1,cols):
            # Dynamic programing -- aka. divide and conquer:
            # Since gap penalties are linear in gap size
            # the score of an alignmet of length l only depends on the   
            # the l-th characters in the alignment (match - mismatch - gap)
            # and the score of the one shorter (l-1) alignment,
            # i.e. we can calculate how to extend an arbritary alignment
            # soley based on the previous score value.  
            
            if i-1 == 0 and j-1 == 0:
                tp = 0
            else:
                num = abs(float(seq2[i-1][0])+TR[i-1][j-1]-float(seq1[j-1][0])-TC[i-1][j-1])
                den = max(float(seq2[i-1][0])+TR[i-1][j-1],float(seq1[j-1][0])+TC[i-1][j-1])
                #temporal penalty function
                tp = T*(num/(den + 1e-8))
                
            choice1 = score[i-1][j-1] + s[(seq1[j-1][1] + seq2[i-1][1])] - tp #diagonal
            choice2 = score[i-1][j] - gap #up
            choice3 = score[i][j-1] - gap #left
            choices = [choice1,choice2,choice3]
            score[i][j] = max(choices)    
            
            # update traceback matrix 0-diagonal, 1-up, 2-left
            traceback[i][j] = choices.index(max(choices))
            
            if traceback[i][j] == 0:
                TR[i][j] = 0
                TC[i][j] = 0
                  
            elif traceback[i][j] == 1:
                TR[i][j] = TR[i-1][j] + float(seq2[i-1][0])
                TC[i][j] = TC[i-1][j]
                
            elif traceback[i][j] == 2:
                TR[i][j] = TR[i][j-1] 
                TC[i][j] = TC[i][j-1] + float(seq1[j-1][0])
                
# deducing the alignment from the traceback matrix
def alignment(traceback,rows,cols,seq1,seq2):

    aseq1 = ''
    aseq2 = ''
    
    #number of aligned events
    count_aligned = 0
    
    #We reconstruct the alignment into aseq1 and aseq2, 
    j = cols-1
    i = rows-1
    while i>0 and j>0:
        
        # going diagonal
        if traceback[i][j] == 0:
            aseq1 = seq1[j-1][1] + aseq1
            aseq2 = seq2[i-1][1] + aseq2
            i -= 1
            j -= 1
            count_aligned = count_aligned + 1
            
        # going up -gap in sequence1 (top one)
        elif traceback[i][j] == 1:
            aseq1 = '_' + aseq1
            aseq2 = seq2[i-1][1] + aseq2
            i -= 1
        # going left -gap in sequence2 (left one)
        elif traceback[i][j] == 2:
            aseq1 = seq1[j-1][1] + aseq1
            aseq2 = '_' + aseq2
            j -= 1
        else:
            #should never get here..
            print('ERROR')
            i=0
            j=0
            aseq1='ERROR';aseq2='ERROR';seq1='ERROR';seq2='ERROR'
    
    while i>0:
        #If we hit j==0 before i==0 we keep going in i (up).
        aseq1 = '_' + aseq1
        aseq2 = seq2[i-1][1] + aseq2
        i -= 1     

    while j>0:
        #If we hit i==0 before j==0 we keep going in j (left).
        aseq1 = seq1[j-1][1] + aseq1
        aseq2 = '_' + aseq2
        j -= 1
        
        
    aligned = [aseq1, aseq2,count_aligned]
    
    return aligned
    
# MAIN ALGORITHM
#INPUT:
    #df_encoded :  a dataframe containing N elements and 2 columns, one for the id and other
    #for the temporal sequences ('id_patient' & 'aux_encode')
    
    #gap: gap penalty used in the TNW algorithm
    
    #T: #user-defined heuristic  and represents the maximum penalty that will be imposed on S(xi,yi) 
    #for temporal differences
    
    #s: pre-defined scoring system
    
    #normalized: variable that takes value 0 or 1, where 0 indicates there is no normalization of the scores
    # and 1 we normalize our scores dividing the scores by the number of aligned events
    
#OUTPUT:
    #results : a dataframe resuming all the information regarding the perfomed alignments
    #it presents the id of the elements being aligned, the correspodent original sequences, 
    # the aligned sequences and final score
    
def main_algorithm(df_encoded,gap,T,s,normalized):
    
    #get all the possible combinations between the patients to perform alignment
    patient_comb = list(itertools.combinations(df_encoded['id_patient'],2))
    
    #set id_patient as index column it will be helpful for later manipulation of sequences
    df_encoded.set_index('id_patient',inplace=True)
    
    results = pd.DataFrame(patient_comb,columns = ['patient1','patient2'])
    list_sequences_aligned = []
    list_scores = []
    list_sequences = []
    
    #analyze every possible combination between patints
    for patient_pair in patient_comb:
        #get the sequences to be aligned
        seq1_encoded = df_encoded.loc[patient_pair[0],'aux_encode']
        seq2_encoded = df_encoded.loc[patient_pair[1],'aux_encode']
    
        list_sequences.append([seq1_encoded,seq2_encoded])
         #split the sequences
        aux1 = seq1_encoded.split(",")
        aux2 = seq2_encoded.split(",")

        seq1 = []
        seq2 = []
        
        for seq in aux1:
            seq1.append(seq.rsplit(".",1))
    
        for seq in aux2:
            seq2.append(seq.rsplit(".",1))
        
        cols=len(seq1)+1
        rows=len(seq2)+1
                
        score = score_initialisation(rows,cols,gap)
        traceback = traceback_initialisation(rows,cols)
        TR = TR_initialisation(rows,cols,traceback,seq2)
        TC = TC_initialisation(rows,cols,traceback,seq1)
        
        calculate_scores(score,traceback,rows,cols,seq1,seq2,TR,TC,gap,T,s)
        sequences_aligned = alignment(traceback,rows,cols,seq1,seq2)
        
        list_sequences_aligned.append(sequences_aligned)
        
        if(normalized == 1):
            #normalization of scores
            list_scores.append((score[rows-1][cols-1])/sequences_aligned[2])
        else:
            list_scores.append(score[rows-1][cols-1])
        
    results['sequences'] = pd.Series(list_sequences)
    results['sequences_aligned'] = pd.Series(list_sequences_aligned)
    results['score'] = pd.Series(list_scores)
    return results

    
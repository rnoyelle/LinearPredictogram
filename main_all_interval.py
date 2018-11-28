 
"""
select sample -600 à 500 + 600
+
select match -600 à +600

with stim

"""
import sys
import os
import os.path

import lib.cnn.matnpyio as io
#import lib.cnn.cnn as cnn 
import lib.cnn.matnpy as matnpy

import tensorflow as tf
import numpy as np
from math import ceil

import random
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold

import pandas as pd
import datetime


import lib.cnn.preprocess as pp


from sklearn.linear_model import LinearRegression

import os


################################################
#################### PARAMS ####################
################################################

#################
### base path ###
#################

base_path = '/media/rudy/disk2/lucy/'

###################
### data params ###
###################

session = os.listdir(base_path)
session.remove('unique_recordings.mat')
print(session)
# '150210'
for sess_no in['150427',
 '150504',
 '150416',
 '150428',
 '150430',
 '150413',
 '150415',
 '150528',
 '150414',
 '150511',
 '150512',
 '150403',
 '150527',
 '150429',
 '150608',
 '150529']: #session :  #session : # '150128', 150210 also a good session
    print(sess_no)
    seed = np.random.randint(1,10000)

    # path
    raw_path = base_path +sess_no+'/session01/'
    rinfo_path = base_path +sess_no+'/session01/' + 'recording_info.mat'

    only_correct_trials = True

    #align_on, from_time, to_time = 'sample', 0, 500 
    #lowcut, highcut, order = 30, 300, 3


    align_on1, from_time1, to_time1 = 'sample', -800, 500 + 600 
    align_on1_match, from_time1_match, to_time1_match = 'match', -600, 2000 

    for lowcut1, highcut1, order1 in [[7,12,3]] : #[[7,12,3], [12,30,3]] : #= 8, 14, 3
        for lowcut2, highcut2, order2 in [[7,12,3]] :# [[7,12,3], [12,30,3]] :
            
            print(lowcut1, highcut1, order1)

            align_on2, from_time2, to_time2 = align_on1, from_time1, to_time1 # 
            align_on2_match, from_time2_match, to_time2_match = align_on1_match, from_time1_match, to_time1_match
            #lowcut2, highcut2, order2 = lowcut1, highcut1, order1

            window_size1 = 200
            window_size2 = window_size1 #window_size1 ### 0.72
            step = 100
            delay = 0


            select_elec_by = 'cortex' # 'areas' or 'cortex'

            # if select_elec_by == 'areas':
            #     areas1 = ['DP']
            #     num1 = 3
            #     areas2 = ['V1']
            #     num2 = 1
            #     cortex1='Prefontal'
            #     cortex2='Prefontal'
            # elif select_elec_by == 'cortex':
            for cortex1 in ['Visual', 'Prefontal', 'Motor', 'Parietal']:#['Visual', 'Prefontal', 'Motor', 'Somatosensory', 'Parietal'] : #= 'Visual' # coding <U16
                for cortex2 in ['Parietal'] : #cortex2 = 'Visual' # 
                    print(cortex1, cortex2)

                    areas1 = io.get_area_cortex(rinfo_path, cortex1, unique = True)
                    areas2 = io.get_area_cortex(rinfo_path, cortex2, unique = True)
                    if len(areas1) == 0 or len(areas2) == 0:
                        #could happend for some session in Somatosensory
                        continue



                    # print(areas1)
                    # print(areas2)

                    renorm = True # si les données doivent être normées ou non (centrées réduites pour les entrées et les sorties indépendement) 

                        
                    #######################
                    ### LEARNING PARAMS ###
                    #######################

                    train_size = 0.8
                    test_size = 1 - train_size
                    #seed = np.random.randint(1,10000)


                    ##################################################
                    #                    GET DATA                    #
                    ##################################################

                    data1, data2, area_names1, area_names2, time_step, stim = matnpy.get_subset_by_areas(sess_no, raw_path, 
                                                                                                                align_on1, from_time1, to_time1, 
                                                                                                                window_size1, 
                                                                                                                lowcut1, highcut1, areas1,
                                                                                                                align_on2, from_time2, to_time2, 
                                                                                                                window_size2, 
                                                                                                                lowcut2, highcut2, 
                                                                                                                areas2,
                                                                                                                step, delay,
                                                                                                                epsillon = 26, order = 3,
                                                                                                                only_correct_trials = only_correct_trials, renorm = renorm )



                        

                    data1_match, data2_match, area_names1, area_names2, time_step_match, stim_match = matnpy.get_subset_by_areas(sess_no, raw_path, 
                                                                                                                align_on1_match, from_time1_match, to_time1_match, 
                                                                                                                window_size1, 
                                                                                                                lowcut1, highcut1, areas1,
                                                                                                                align_on2_match, from_time2_match, to_time2_match, 
                                                                                                                window_size2, 
                                                                                                                lowcut2, highcut2, 
                                                                                                                areas2,
                                                                                                                step, delay,
                                                                                                                epsillon = 26, order = 3,
                                                                                                                only_correct_trials = only_correct_trials, renorm = renorm )



                        


                    classes = 5
                    n_chans1 = data1.shape[1]
                    samples_per_trial1 = data1.shape[2]

                    n_chans2 = data2.shape[1]
                    samples_per_trial2 = data2.shape[2]

                    ##### CONCAT DATA sample et data match

                    data1_all = np.concatenate((data1, data1_match), axis=0)
                    data2_all = np.concatenate((data2, data2_match), axis=0)
                    stim = np.concatenate((stim, stim_match), axis=0)
                    time_step = np.concatenate((time_step, time_step_match + time_step[-1]+step), axis = 0)



                    classes = 5
                    n_chans1 = 1 # data1.shape[1]
                    samples_per_trial1 = data1_all.shape[2]

                    n_chans2 = 1 #data2.shape[1]
                    samples_per_trial2 = data2_all.shape[2]
                    
                    
                    if data1_all.shape[0] == 0 or data2_all.shape[0] == 0:
                        break
                    
                    print(lowcut1, highcut1, order1)
                    print(lowcut2, highcut2, order2)
                    
                    print(cortex1)
                    print(cortex2)
                    print(areas1)
                    print(areas2)

                    for area1 in areas1 :
                        idx1 = []
                        for count, area in enumerate(area_names1):
                            if area == area1 :
                                idx1.append(count)
                                
                        for count1, idx_channel1 in enumerate(idx1):
                            
                            for area2 in areas2 :
                                idx2 = []
                                for count, area in enumerate(area_names2):
                                    if area == area2 :
                                        idx2.append(count)
                                        
                                for count2, idx_channel2 in enumerate(idx2):
                                    
                                        data1 = data1_all[:,[idx_channel1],:,:]
                                        data2 = data2_all[:,[idx_channel2],:,:]
                                        
                                        #print(data1.shape)
                                        
                                        if renorm == True :
                                            data1 = pp.renorm(data1)
                                            data2 = pp.renorm(data2)
                                        print(area1, count1, area2, count2)
                                        
                                        #data1 = data1[:,0,:,0] # data1 = np.reshape(data1, (data1.shape[0], -1))
                                        #data2 = data2[:,0,:,0] # data2 = np.reshape(data2, (data2.shape[0], -1))
                                        
                                        data1 = np.reshape(data1, (data1.shape[0], -1))
                                        data2 = np.reshape(data2, (data2.shape[0], -1))
                                        
                                        ################################################
                                        #         TRAINING AND TEST NETWORK            #
                                        ################################################
                                        
                                        ### SPLIT
                                        indices = [i for i in range(data1.shape[0])]
                                        
                                        
                                        targets = [ (time_step[i]/step) * classes + np.argmax(stim[i]) for i in range(data1.shape[0]) ]
                                        
                                        x_train, x_test, y_train, y_test, ind_train, ind_test = (
                                            train_test_split(
                                                data1, 
                                                data2, 
                                                indices,
                                                test_size=test_size, 
                                                random_state=seed,
                                                stratify = targets
                                                )
                                            ) 
                                            
                                        x_stim_test = stim[ind_test, :]
                                        
                                        ### TRAIN linear regression
                                        reg = LinearRegression().fit(x_train, y_train)
                                        
                                        # test linear regression
                                        
                                        # on trainning base
                                        r2_train = reg.score(x_train, y_train)
                                        # on testing base
                                        r2_test = reg.score(x_test, y_test)
                                        
                                        # error bar                                                                                
                                        y_test_predict = reg.predict(x_test)
                                        mse_test = np.mean( (y_test_predict - y_test)**2, axis =(1) ) # y_test.shape = (N_samples, N_features) with n_features  = n_time *channel
                                        r2_error_bar_test = ( np.std(mse_test)/np.sqrt(y_test.shape[0]) )/np.var(y_test)
                                        

                                        # time_detail                                        
                                        r2_train_time = []
                                        r2_error_bar_train_time = []
                                        step_train = time_step[ind_train]
                                        for step_loop in np.unique(step_train):
                                            index = (step_train==step_loop)
                                            r2_train_time.append( reg.score(x_train[index], y_train[index]) )
                                                                                        #error bar
                                            y_train_predict_time = reg.predict(x_train[index])
                                            mse_train_time = np.mean( (y_train[index] - y_train_predict_time)**2, axis =(1) ) # y_test.shape = (N_samples, N_features) with n_features  = n_time * channel
                                            r2_error_bar_train_time.append(( np.std(mse_train_time)/np.sqrt(y_train_predict_time.shape[0]) )/np.var(y_train[index]) )
                                            
                                            
                                        
                                        
                                        
                                        r2_test_time = []
                                        r2_error_bar_test_time = []
                                        step_test = time_step[ind_test]
                                        for step_loop in np.unique(step_test):
                                            index = (step_test == step_loop)
                                            r2_test_time.append( reg.score(x_test[index], y_test[index]) )
                                            
                                            #error bar
                                            y_test_predict_time = reg.predict(x_test[index])
                                            mse_test_time = np.mean( (y_test[index] - y_test_predict_time)**2, axis =(1) ) # y_test.shape = (N_samples, N_features) with n_features  = n_time * channel
                                            r2_error_bar_test_time.append(( np.std(mse_test_time)/np.sqrt(y_test_predict_time.shape[0]) )/np.var(y_test[index]) )
                                            
                                            
                                            
                                            
                                        step_list = []
                                        for step_loop in np.unique(step_test):
                                            step_list.append(step_loop)
                                            
                                        where_first_part_end = 1+ int( (to_time1 - from_time1 - window_size1)/step )
                                        step_list = np.array(step_list)
                                        step_list[:where_first_part_end] = step_list[:where_first_part_end]  + from_time1 
                                        step_list[where_first_part_end:] = step_list[where_first_part_end:]  + from_time1 + 2000 - where_first_part_end*step # 2000 = 500 + max( time of delay)                                         
                                        step_list = step_list + int(window_size1/2)
                                        step_list = list(step_list)
                                            
                                            
                            
                                        str_freq1 = 'low'+str(lowcut1)+'high'+str(highcut1)+'order'+str(order1)
                                        str_freq2 = 'low'+str(lowcut2)+'high'+str(highcut2)+'order'+str(order2)


                                        data_tuning = [ sess_no, area1, count1, area2, count2,   
                                                        cortex1, cortex2,  
                                                        str_freq1, str_freq2,  
                                                        window_size1, window_size2, 
                                                        step, delay,  
                                                        len(ind_test), len(ind_train), 
                                                        n_chans1, n_chans2, 
                                                        only_correct_trials, 
                                                        r2_train, r2_test,
                                                        r2_train_time, r2_test_time,
                                                        r2_error_bar_test, r2_error_bar_test_time,
                                                        step_list,
                                                        renorm, seed] 


                                        df = pd.DataFrame([data_tuning],
                                                        columns=[ 'session', 'area1','num1', 'area2', 'num2',   
                                                        'cortex1', 'cortex2',  
                                                        'str_freq1', 'str_freq2',  
                                                        'window_size1', 'window_size2', 
                                                        'step', 'delay',  
                                                        'len(ind_test)', 'len(ind_train)', 
                                                        'n_chans1', 'n_chans2', 
                                                        'only_correct_trials', 
                                                        'r2_train', 'r2_test', 
                                                        'r2_train_time', 'r2_test_time',
                                                        'r2_error_bar_test', 'r2_error_bar_test_time',
                                                        'step_list',
                                                        'renorm', 'seed'] ,
                                                        index=[0])


                                        file_name = '/home/rudy/Python2/regression_linear/result2/' + 'result_sess_no_'+str(sess_no)+'channel_to_channel_all_interval_with_error_bar.csv'
                                        file_exists = os.path.isfile(file_name)
                                        if file_exists :
                                            with open(file_name, 'a') as f:
                                                df.to_csv(f, mode ='a', index=False, header=False)
                                        else:
                                            with open(file_name, 'w') as f:
                                                df.to_csv(f, mode ='w', index=False, header=True)




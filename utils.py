import os
import re
import datetime
import time
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def save_model_and_weights(experiment_number,model,model_type):
    exp_path = "ExperimentsWeather/"+model_type+"/Experiment" + str(experiment_number)
    check_point_path = exp_path+"/checkpoints"
    os.mkdir(check_point_path)
    try:
        model_path = exp_path+"/model{}.h5".format(experiment_number)
        model.save(model_path,save_format="h5")
    except Exception as e:
        print("Could not save the model in h5 format")
        print("Exception error: ",str(e))
    try:
        model_path = exp_path+"/model{}_tf".format(experiment_number) 
        model.save(model_path,save_format="tf")
    except Exception as e2:
        print("Could not save the model in tf format")
        print("Exception error: ",str(e2))
    try:
        check_point_path = exp_path+'/checkpoints/checkpoint.hdf5'
        model.save_weights(check_point_path)
    except Exception as e:
        print("Could not save the model weights in h5 format")
        print("Exception error: ",str(e))
    try:
        checkpoint_path = exp_path+"/checkpoints/checkpoint"
        model.save_weights(checkpoint_path)
    except Exception as e2:
        print("Could not save the model weights in checkpoint format")
        print("Exception error: ",str(e2))

def create_info_training_file(experiment_number,model_type):
    filename = "ExperimentsWeather/"+model_type+"/Experiment"+str(experiment_number)+"/infos_training"+str(experiment_number)+".txt"
    with open(filename, "w") as file:
        file.write("")

def get_experiment_number(model_type):
    experiments_folders_list = os.listdir(path='ExperimentsWeather/'+model_type)
    if(len(experiments_folders_list) == 0): #empty folder
        return 1
    else:  
        temp_numbers=[]
        for folder in experiments_folders_list:
            number = re.findall(r'\d+', folder)
            if(len(number)>0):
                temp_numbers.append(int(number[0]))
        return max(temp_numbers) + 1
    
def create_experiment_folder(experiment_number,model_type):
    try:
        path_new_experiment = "ExperimentsWeather/"+model_type+"/Experiment" + str(experiment_number)
        os.mkdir(path_new_experiment)
    except Exception as e:
        print ("Creation of the directory {} failed".format(path_new_experiment))
        print("Exception error: ",str(e)) 
        
def create_main_experiment_folder():
    if(not os.path.isdir("ExperimentsWeather")):
        try:
            os.mkdir("ExperimentsWeather")
        except Exception as e:
            print ("Creation of the main experiment directory failed")
            print("Exception error: ",str(e))
            
def create_model_folder(model_type):
    path_model = "ExperimentsWeather/"+model_type
    if(not os.path.isdir(path_model)):
        try:
            os.mkdir(path_model)
        except Exception as e:
            print ("Creation of the main model experiment directory failed")
            print("Exception error: ",str(e))   

def list_dates(start,end):
    """ This creates a list of of dates between the 'start' date and the 'end' date """
    # create datetime object for the start and end dates
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    # generates list of dates between start and end dates
    step = datetime.timedelta(days=1)
    dates = []
    while start <= end:
        dates.append(start.date())
        start += step
    # return the list of dates in string format
    return [str(date) for date in dates]

def get_dates():      
    start_date = datetime.datetime(2005, 5, 4)
    end_date = datetime.datetime(2020, 4, 24)
    dates = list_dates(str(start_date).split(" ")[0],str(end_date).split(" ")[0])
    return dates

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

selected_cities = ["london","dublin","paris","munich",
                   "nice","marseille","lyon","barcelona",
                   "malaga","luxembourg","frankfurt",
                   "brussels","maastricht","rotterdam","amsterdam",
                   "hamburg","berlin","copenhagen"]

city_to_index = {"london":0,
                 "dublin":1,
                 "paris":2,
                 "munich":3,
                 "nice":4,
                 "marseille":5,
                 "lyon":6,
                 "barcelona":7,
                 "malaga":8,
                 "luxembourg":9,
                 "frankfurt":10,
                 "brussels":11,
                 "maastricht":12,
                 "rotterdam":13,
                 "amsterdam":14,
                 "hamburg":15,
                 "berlin":16,
                 "copenhagen":17}

index_to_city = {
                 0:"london",
                 1:"dublin",
                 2:"paris",
                 3:"munich",
                 4:"nice",
                 5:"marseille",
                 6:"lyon",
                 7:"barcelona",
                 8:"malaga",
                 9:"luxembourg",
                 10:"frankfurt",
                 11:"brussels",
                 12:"maastricht",
                 13:"rotterdam",
                 14:"amsterdam",
                 15:"hamburg",
                 16:"berlin",
                 17:"copenhagen"
                 }

feature_to_index = {'high_temp(F)':0,
                    'low_temp(F)':1,
                    'avg_temp(F)':2,
                    'dewpoint(F)':3,
                    'high_dew(F)':4,
                    'low_dew(F)':5,
                    'avg_dew(F)':6,
                    'maxwindspeed(mph)':7,
                    'visibility(mi)':8,
                    'sea_level_pressure(Hg)':9,
                    'temp_obs(F)':10,
                    'dewpoint_obs(F)':11,
                    'humidity(%)':12,
                    'wind':13,
                    'wind_speed(mph)':14,
                    'wind_gust(mph)':15,
                    'pressure(in)':16,
                    'condition':17
                    }


def get_number_features(cities):
    first_city = cities[0]
    df = pd.read_csv("data/"+first_city+".csv")
    return df.shape[1] - 1#not the date

def get_num_dates():
    dates = get_dates()
    return len(dates)

network_window_size = 10
depth = 1
window_size = network_window_size * depth
number_dates = get_num_dates()

#@timeit
def get_indexes_input(network_window_size,depth,window_size,total_number_dates):
    step_size = 1
    idx = [i for j in range(window_size-1,total_number_dates,step_size) for i in range(j-(window_size-1),j+1,1)]
    assert len(idx)%window_size == 0, "Warning: due to the window size and step_size, data will be truncated, and {} dates won't be used".format(len(idx)%window_size)
    num_rows = len(idx)//window_size
    big_matrix = np.zeros((num_rows,window_size),dtype=int)
    for i in range(num_rows):
        big_matrix[i] = idx[i*window_size:(i*window_size)+window_size]
    return big_matrix,num_rows

#@timeit
def get_x(scaled_volume,network_window_size,number_features,number_cities,depth,number_steps_ahead,big_matrix_indexes,num_rows_input):    
    input_indexes = big_matrix_indexes[:-number_steps_ahead]#we discard the 2 last rows
    x = []
    for k in range(network_window_size):
        column_input = np.zeros((num_rows_input,number_features,number_cities,depth))
        for j in range(num_rows_input):
            try:
                indexes_to_take = input_indexes[j,k*depth:(k*depth)+depth]
                cube = scaled_volume[:,:,indexes_to_take]
                column_input[j,:,:,:] = cube
            except Exception as e:
                print("j={},k={}".format(j,k))
                print("Exception error:",e)
                sys.exit()
        x.append(column_input)
    return x

def get_indexes_output(big_matrix,num_rows,number_steps_ahead,big_matrix_indexes,number_target_cities):
    output_indexes = np.zeros((num_rows-number_steps_ahead,number_target_cities),dtype=int)
    for i in range(number_steps_ahead,num_rows):
        output_indexes[i-number_steps_ahead] = big_matrix_indexes[i,-1]
    return output_indexes

#@timeit
def get_y(scaled_volume,big_matrix_indexes,num_rows,target_feature,target_cities,number_steps_ahead,number_target_cities):
    output_indexes = get_indexes_output(big_matrix_indexes,num_rows,number_steps_ahead,big_matrix_indexes,number_target_cities)
    y = np.zeros(output_indexes.shape)
    for i in range(len(output_indexes)):
        indexes_to_take = output_indexes[i]

        for j in range(len(target_cities)):
            y[i][j] = scaled_volume[:,:,indexes_to_take[j]][target_feature][target_cities[j]]
    return y

def get_merged_big_volume(cities,number_features,number_cities,len_dates):
    cities_list = []
    matrix = np.zeros((18,18))
    volume = np.zeros((number_features,number_cities,len_dates))
    for city in cities:
        df = pd.read_csv("data/"+city+".csv")
        df_numpy = df.to_numpy()
        cities_list.append(df_numpy)
    for i in range(number_dates):  
        for j in range(number_cities):
            matrix[:,j] = cities_list[j][i][1:]
        volume[:,:,i] = matrix
    return volume

def tensor_to_matrix(tensor): 
    len_dates = tensor.shape[2]
    number_cities = tensor.shape[1]
    number_features = tensor.shape[0]
    matrix_for_scaling = np.ones((len_dates,number_cities*number_features))
    for i in range(number_cities):
        for j in range(len_dates):
            features_city = tensor[:,i,j]
            matrix_for_scaling[j,i*18:(i+1)*18] = features_city        
    return matrix_for_scaling
    
def scale_matrix(matrix_for_scaling):
    scaling_infos = []
    number_cities = 18
    number_features = 18
    scaled_matrix = np.zeros(matrix_for_scaling.shape)    
    for i in range(number_cities*number_features):
        s = MinMaxScaler(feature_range=(0, 1))
        temp = s.fit_transform(matrix_for_scaling[:,i].reshape(-1,1))
        temp = np.squeeze(temp,-1)
        scaled_matrix[:,i] = temp
        scaling_infos.append(s)        
    return scaled_matrix,scaling_infos


#function to transform a matrix of size (dates,cities*features) to a tensor of shape (features,cities,dates)    
def matrix_to_tensor(matrix):
    len_dates = matrix.shape[0]
    number_cities = 18
    number_features= 18
    new_tensor = np.ones((number_features,number_cities,len_dates))      
    for i in range(len_dates):
        for j in range(number_cities):
            features = matrix[i,j*number_features:(j+1)*number_features]
            new_tensor[:,j,i] = features
    return new_tensor

def scale_volume(volume):
    matrix = tensor_to_matrix(volume)
    scaled_matrix, scaling_info_list = scale_matrix(matrix)
    scaled_volume = matrix_to_tensor(scaled_matrix)
    return scaled_volume,scaling_info_list

def get_dataset_model1(number_steps_ahead,target_feature_name,target_cities_names):
    x_model_2_train,y_train,x_model_2_valid,y_valid,x_model_2_test,y_test,scaling_infos = get_dataset_model2(number_steps_ahead,target_feature_name,target_cities_names)
    x_model_1_train = np.squeeze(x_model_2_train,axis=-1)
    x_model_1_valid = np.squeeze(x_model_2_valid,axis=-1)
    x_model_1_test = np.squeeze(x_model_2_test,axis=-1)
    
    x_train_m1_permuted = np.transpose(x_model_1_train,(0,2,1,3))
    x_train_m1_permuted = np.transpose(x_train_m1_permuted,(0,1,3,2))
    
    x_valid_m1_permuted = np.transpose(x_model_1_valid,(0,2,1,3))
    x_valid_m1_permuted = np.transpose(x_valid_m1_permuted,(0,1,3,2))
    
    x_test_m1_permuted = np.transpose(x_model_1_test,(0,2,1,3))
    x_test_m1_permuted = np.transpose(x_test_m1_permuted,(0,1,3,2))
    
    return x_train_m1_permuted,y_train,x_valid_m1_permuted,y_valid,x_test_m1_permuted,y_test,scaling_infos

def get_dataset_model2(number_steps_ahead,target_feature_name,target_cities_names):
    target_feature = feature_to_index[target_feature_name]
    target_cities = []
    for city_name in target_cities_names:
        target_cities.append(city_to_index[city_name])
    number_cities = 18
    number_features = 18
    number_target_cities = 6
    volume = get_merged_big_volume(selected_cities,number_features,number_cities,number_dates)
    scaled_volume,scaling_infos = scale_volume(volume)
    big_matrix_indexes,num_rows = get_indexes_input(network_window_size,depth,window_size,number_dates)
    num_rows_input = num_rows-number_steps_ahead
    total_x = get_x(scaled_volume,network_window_size,number_features,number_cities,depth,number_steps_ahead,big_matrix_indexes,num_rows_input)
    lags = len(total_x)
    total_y = get_y(scaled_volume,big_matrix_indexes,num_rows,target_feature,target_cities,number_steps_ahead,number_target_cities)
    end_train = int(0.8*total_x[0].shape[0])
    start_valid = end_train
    end_valid = start_valid + int(0.1*total_x[0].shape[0])
    end_train_valid = end_valid
    start_test = end_valid
    test_samples = total_x[0].shape[0] - int(0.8*total_x[0].shape[0]) - int(0.1*total_x[0].shape[0])
     
    x_train_and_valid = []
    for item in total_x:
        x_train_and_valid.append(item[0:end_train_valid,:,:,:])
    y_train_and_valid = total_y[0:end_train_valid]
    *x_train_and_valid,y_train_and_valid = shuffle(*x_train_and_valid,y_train_and_valid,random_state=42)
    
    x_model_2_train_and_valid = np.zeros((x_train_and_valid[0].shape[0],lags,18,18,1))
    for i in range(x_train_and_valid[0].shape[0]):
        for j in range(lags):
            x_model_2_train_and_valid[i,j,:,:,:] = x_train_and_valid[j][i]
        
    x_model_2_train = x_model_2_train_and_valid[0:end_train]
    y_train = y_train_and_valid[0:end_train]
    
    x_model_2_valid = x_model_2_train_and_valid[start_valid:end_valid]
    y_valid = y_train_and_valid[start_valid:end_valid]    
    
    x_model_2_test = np.zeros((test_samples,lags,18,18,1))
    y_test = total_y[start_test:]
    for i in range(test_samples):
        for j in range(lags):
            x_model_2_test[i,j,:,:,:] = total_x[j][i+start_test]
    
    return x_model_2_train,y_train,x_model_2_valid,y_valid,x_model_2_test,y_test,scaling_infos

def get_dataset_model3(number_steps_ahead,target_feature_name,target_cities_names):
    target_feature = feature_to_index[target_feature_name]
    target_cities = []
    for city_name in target_cities_names:
        target_cities.append(city_to_index[city_name])
    number_cities = 18
    number_features = 18
    number_target_cities = 6
    volume = get_merged_big_volume(selected_cities,number_features,number_cities,number_dates)
    scaled_volume,scaling_infos = scale_volume(volume)
    big_matrix_indexes,num_rows = get_indexes_input(network_window_size,depth,window_size,number_dates)
    num_rows_input = num_rows-number_steps_ahead
    total_x = get_x(scaled_volume,network_window_size,number_features,number_cities,depth,number_steps_ahead,big_matrix_indexes,num_rows_input)
    total_y = get_y(scaled_volume,big_matrix_indexes,num_rows,target_feature,target_cities,number_steps_ahead,number_target_cities)

    end_train = int(0.8*total_x[0].shape[0])
    start_valid = end_train
    end_valid = start_valid + int(0.1*total_x[0].shape[0])
    end_train_valid = end_valid
    start_test = end_valid   
    
    x_train_and_valid = []
    for item in total_x:
        x_train_and_valid.append(item[0:end_train_valid,:,:,:])
    y_train_and_valid = total_y[0:end_train_valid]
    *x_train_and_valid,y_train_and_valid = shuffle(*x_train_and_valid,y_train_and_valid,random_state=42)
    
    x_train = []
    for item in x_train_and_valid:
        x_train.append(item[0:end_train,:,:,:])
    y_train = y_train_and_valid[0:end_train]
    x_train_dict = {}
    for i in range(network_window_size):
        x_train_dict["input"+str(i+1)] = x_train[i]
        
    x_valid = []
    for item in x_train_and_valid:
        x_valid.append(item[start_valid:end_valid,:,:,:])
    y_valid = y_train_and_valid[start_valid:end_valid]
    x_valid_dict = {}
    for i in range(network_window_size):
        x_valid_dict["input"+str(i+1)] = x_valid[i] 
    
    x_test = []
    for item in total_x:
        x_test.append(item[start_test:,:,:,:])
    y_test = total_y[start_test:]
    x_test_dict = {}
    for i in range(network_window_size):
        x_test_dict["input"+str(i+1)] = x_test[i]
        
    return x_train_dict,y_train,x_valid_dict,y_valid,x_test_dict,y_test,scaling_infos

def prepare_data_for_model_4(data_dict):
    output_dict = {}
    temp = np.zeros((2,18,18))
    for i in range(5):
        output_dict["input"+str(i+1)] = np.zeros((data_dict["input1"].shape[0],2,18,18,1))
    for i in range(data_dict["input1"].shape[0]):
        for j in range(0,5):
            first_element = np.squeeze(data_dict["input"+str(j*2+1)][i],-1)
            second_element = np.squeeze(data_dict["input"+str(j*2+2)][i],-1)
            temp[0] = first_element
            temp[1] = second_element
            temp2 = np.expand_dims(temp,-1)
            output_dict["input"+str(j+1)][i] = temp2

    return output_dict

def get_dataset_model4(number_steps_ahead,target_feature_name,target_cities_names):
    x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = get_dataset_model3(number_steps_ahead,target_feature_name,target_cities_names)
    
    x_train_dict_m4 = prepare_data_for_model_4(x_train)
    x_valid_dict_m4 = prepare_data_for_model_4(x_valid)
    x_test_dict_m4 = prepare_data_for_model_4(x_test)
    
    return x_train_dict_m4,y_train,x_valid_dict_m4,y_valid,x_test_dict_m4,y_test,scaling_infos
    
def get_folders_started(model_type):
    create_main_experiment_folder()      
    create_model_folder(model_type)
    experiment_number = get_experiment_number(model_type)
    create_experiment_folder(experiment_number,model_type)
    base_filenames = model_type +  str(experiment_number)
    create_info_training_file(experiment_number,model_type)
    return base_filenames,experiment_number
    
def get_rescaled_data(scaling_infos,feature_name,city_names,y_test,y_predicted,y_train):
    target_feature = feature_to_index[feature_name]
    predictions_per_city = {}
    for city_name in city_names:        
        target_city = city_to_index[city_name]
        scaling_index = target_city*18 +target_feature
        scaling_infos_target_dict = scaling_infos[scaling_index].__dict__
        
        target_scale = scaling_infos_target_dict["scale_"][0]
        target_min = scaling_infos_target_dict["min_"][0]
        
        y_test_np = np.array(y_test)
        y_test_rescaled = y_test_np - target_min
        y_test_rescaled /= target_scale
        
        y_predicted_rescaled = y_predicted - target_min
        y_predicted_rescaled /= target_scale
                
        y_train_rescaled = y_train - target_min
        y_train_rescaled /= target_scale
        predictions_per_city[target_city] = (y_test_rescaled,y_predicted_rescaled,y_train_rescaled)
    return predictions_per_city

def plot_actual_vs_prediction(target_cities_name,target_feature_name,days_ahead,predictions_per_city_dict,model_type,experiment_number,base_filenames):
    alias_cities = []
    for city in target_cities_name:
        alias_cities.append("".join(char for char in city[0:3]))
    alias_string = "_".join(elem for elem in alias_cities)    
    rows = 2#hardcoded
    cols = 3#hardcoded
    fig, axs = plt.subplots(rows,cols, figsize=(40, 21))
    fig.subplots_adjust(left=0, bottom=0.5, right=0.5, top=1.5, wspace=0, hspace=None)
    axs = axs.ravel()
    for i in range(rows*cols):
        target_city = city_to_index[target_cities_name[i]]
        y_test_rescaled = predictions_per_city_dict[target_city][0]
        y_predicted_rescaled = predictions_per_city_dict[target_city][1]
        axs[i].set_title(target_cities_name[i]+","+target_feature_name)
        axs[i].plot(y_test_rescaled[:,i], label='Original, {} days ahead'.format(days_ahead))
        axs[i].plot(y_predicted_rescaled[:,i], label='Predicted, {} days ahead'.format(days_ahead))
        axs[i].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
            
    output_filename2 = "ExperimentsWeather/"+model_type+"/Experiment"+str(experiment_number)+"/"+base_filenames+"_"+alias_string+"_"+target_feature_name+"_"+str(days_ahead)+"_actual_vs_prediction.png"
    try:
        fig.savefig(output_filename2,dpi=100)
    except Exception as e:
        print("couldn't save in the experiment folder, saving in the current folder")
        print("reason: ",e)
        filename_output = alias_string+"_"+target_feature_name+"_"+str(days_ahead)+"_actual_vs_prediction.png"
        fig.savefig(filename_output,dpi=100)

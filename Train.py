import utils
from tensorflow.keras.optimizers import Adam
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error
from models import get_att_unistream_model,att_multistream
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',type=int,help="Please choose the number of \
                    epochs, by default 1 epoch", default=1)
                    
args = parser.parse_args()

num_epochs = args.epochs
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)
params = 0

cities = ["paris","london","luxembourg","brussels","frankfurt","rotterdam"]
comma_separated_cities = ",".join(item for item in cities)
alias_cities = []
for item in cities:
    alias_cities.append("".join(char for char in item[0:3]))
alias_string = "_".join(elem for elem in alias_cities) 
model_list = ["att_unistream","att_multistream"]
features = ["wind_speed(mph)","avg_temp(F)"]
steps_ahead_list=[2,4,6]
for model_type in model_list:
    for feature_type in features:
        for step_ahead in steps_ahead_list:
            print("-"*6+" Training configuration: {},{},{},{} ".format(comma_separated_cities,model_type,feature_type,step_ahead)+"-"*6)
            if model_type == "att_unistream":
                x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = utils.get_dataset_model2(step_ahead,feature_type,cities)
                model = get_att_unistream_model()
            elif model_type == "att_multistream":
                x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = utils.get_dataset_model4(step_ahead,feature_type,cities)
                model = att_multistream()
            base_filenames,experiment_number = utils.get_folders_started(model_type)
            for layer in model.layers:
                params_layer = layer.count_params()
                params += params_layer                
            print("The model has {:,} parameters".format(params))            
            model.compile(optimizer = optimizer, loss="mse", metrics=['mse'])
            start_time = time.time()
            history = model.fit(x_train, y_train, validation_data = (x_valid,y_valid),epochs=num_epochs,  verbose=1,batch_size = 16)
            end_time = time.time()
            timespan = end_time - start_time
            print("Timespan training: {}".format(timespan))
            
            y_predicted = model.predict(x_test)
            
            test_mse = mean_squared_error(y_test,y_predicted,multioutput='raw_values')
            mses = []
            test_mae = mean_absolute_error(y_test,y_predicted,multioutput='raw_values')
            maes = []
            
            for i in range(len(cities)):
                print("For city {}, test MSE: {}".format(cities[i],test_mse[i]))
                mses.append(test_mse[i])
                
            for i in range(len(cities)):
                print("For city {}, test MAE: {}".format(cities[i],test_mae[i]))
                maes.append(test_mae[i])
            
            predictions_per_city = utils.get_rescaled_data(scaling_infos,feature_type,cities,y_test,y_predicted,y_train)
            
            utils.plot_actual_vs_prediction(cities,feature_type,step_ahead,predictions_per_city,model_type,experiment_number,base_filenames)
                            
            mses_rescaled = []
            maes_rescaled = []
            print()
            print()
            for i in range(len(cities)):
                city_index = utils.city_to_index[cities[i]]
                y_test_rescaled = predictions_per_city[city_index][0]
                y_predicted_rescaled = predictions_per_city[city_index][1]
                
                test_mse_rescaled = mean_squared_error(y_test_rescaled,y_predicted_rescaled)
                mses_rescaled.append(test_mse_rescaled)
                
                test_mae_rescaled = mean_absolute_error(y_test_rescaled,y_predicted_rescaled)
                maes_rescaled.append(test_mae_rescaled)
                
                print("For city {}, test MSE rescaled: {}".format(cities[i],test_mse_rescaled))
                print("For city {}, test MAE rescaled: {}".format(cities[i],test_mae_rescaled))
            
            info_file = "ExperimentsWeather/"+model_type+"/Experiment"+str(experiment_number)+"/infos_training"+str(experiment_number)+".txt"
            
            with open(info_file, "a+") as file:
                for i in range(len(cities)):                    
                    file.write("{} test MSE: {}\n".format(cities[i],mses[i]))
                    file.write("{} test MAE: {}\n".format(cities[i],maes[i]))
                    file.write("{} test MSE rescaled: {}\n".format(cities[i],mses_rescaled[i]))
                    file.write("{} test MAE rescaled: {}\n".format(cities[i],maes_rescaled[i]))
                file.write("Model number of parameters: {:,}\n".format(params))
                file.write("Timespan training: {}\n".format(timespan))
                file.write("Optimizer: {}\n".format(str(optimizer)))
                file.write("Learning rate: {}\n".format(str(learning_rate)))
                
            utils.save_model_and_weights(experiment_number,model,model_type)
            
            print("-"*6+" End of training of configuration "+"-"*6+"\n\n")
            params = 0
            

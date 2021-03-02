import argparse
import utils
from models import get_conv_plus_lstm,get_ms_conv_plus_lstm_model
from sklearn.metrics import mean_squared_error
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--stepsahead', type=int, help="Please choose the number of steps ahead\
                    (2, 4, or 6), by default 2", default=2)
                    
parser.add_argument('-m','--model', type=str,help="Please choose the type of model \
                    you want to train (att_unistream or att_multistream)",choices=['att_unistream', 'att_multistream'])
                    
parser.add_argument('-f','--feature', type=str,help="Please choose the weather feature \
                    you want to forecast (wind_speed or avg_temperature),by default avg_temperature",choices=['wind_speed', 'avg_temperature'],default="avg_temperature")
                    
args = parser.parse_args()
                    
cities = ["paris","london","luxembourg","brussels","frankfurt","rotterdam"]

if args.feature == "wind_speed":
    feature_type = "wind_speed(mph)"
elif args.feature == "avg_temperature":
    feature_type = "avg_temp(F)"

if args.model == "att_unistream":            
    x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = utils.get_dataset_model2(args.stepsahead,feature_type,cities)
    model = get_conv_plus_lstm()
elif args.model == "att_multistream":
    x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = utils.get_dataset_model4(args.stepsahead,feature_type,cities)
    model = get_ms_conv_plus_lstm_model()
    
model.load_weights("saved_models/{}/{}/{}/checkpoint.hdf5".format(args.model,args.feature,args.stepsahead))
y_predicted = model.predict(x_test)

    

predictions_per_city = utils.get_rescaled_data(scaling_infos,feature_type,cities,y_test,y_predicted,y_train)
for k,v in predictions_per_city.items():
    print(k)
for i in range(len(cities)):
    target_city = utils.city_to_index[cities[i]]
    y_test_rescaled = predictions_per_city[target_city][0]
    y_predicted_rescaled = predictions_per_city[target_city][1]
    test_mse = mean_squared_error(y_test_rescaled,y_predicted_rescaled)
    print("For city {}, test MSE: {}".format(cities[i],test_mse))
    
            
utils.plot_actual_vs_prediction(cities,feature_type,args.stepsahead,predictions_per_city,args.model,0,"")

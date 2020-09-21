import argparse
import utils
from models import get_conv_plus_lstm,get_ms_conv_plus_lstm_model
from sklearn.metrics import mean_squared_error
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--stepsahead', type=int, help="Please choose the number of steps ahead\
                    (2, 4, or 6), by default 2", default=2)
                    
parser.add_argument('-m','--model', type=str,help="Please choose the type of model \
                    you want to train (conv_plus_lstm or ms_conv_plus_lstm)",choices=['conv_plus_lstm', 'ms_conv_plus_lstm'])
                    
parser.add_argument('-f','--feature', type=str,help="Please choose the weather feature \
                    you want to forecast (wind_speed or avg_temperature),by default avg_temperature",choices=['wind_speed', 'avg_temperature'],default="avg_temperature")
                    
args = parser.parse_args()
                    
cities = ["paris","london","luxembourg","brussels","frankfurt","rotterdam"]

if args.model == "conv_plus_lstm":            
    x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = utils.get_dataset_model1(args.stepsahead,args.feature,cities)
    model = get_conv_plus_lstm()
elif args.model == "ms_conv_plus_lstm":
    x_train,y_train,x_valid,y_valid,x_test,y_test,scaling_infos = utils.get_dataset_model3(args.stepsahead,args.feature,cities)
    model = get_ms_conv_plus_lstm_model()
    
model.load_weights("/saved_models/{}/{}/{}/checkpoint.hdf5".format(args.model,args.feature,args.stepsahead))
y_predicted = model.predict(x_test)
test_mse = mean_squared_error(y_test,y_predicted,multioutput='raw_values')
for i in range(len(cities)):
    print("For city {}, test MSE: {}".format(cities[i],test_mse[i]))
    
if args.feature == "wind_speed":
    feature_type = "wind_speed(mph)"
elif args.feature == "avg_temperature":
    feature_type = "avg_temp(F)"
predictions_per_city = utils.get_rescaled_data(scaling_infos,feature_type,cities,y_test,y_predicted,y_train)
            
utils.plot_actual_vs_prediction(cities,feature_type,args.stepsahead,predictions_per_city,args.model,0,"")

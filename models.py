from tensorflow.keras.layers import Conv2D, LSTM, concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input
import tensorflow
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization


def convlstm(lags, features, cities, filters, kernSize):#conv_lstm
    dense1_nodes = 20
    dense2_nodes = 25
    number_target_cities = 6
    input1 = Input(shape = (lags, features, cities, 1))
    block1 = ConvLSTM2D(filters, (kernSize, kernSize), padding = 'same', return_sequences=True, activation='relu',data_format="channels_last")(input1)
    block1 = BatchNormalization()(block1)
    block2 = Flatten(name = 'flatten')(block1)
    block3 = Dense(dense1_nodes, activation='relu')(block2)
    block3 = Dense(dense2_nodes, activation='relu')(block3)
    output1 = Dense(number_target_cities, activation='linear')(block3)
    return Model(inputs=input1, outputs=output1)

def get_convlstm_model():#get_conv_lstm_model
    lags = 10
    features = 18 
    cities = 18 
    filters = 5
    kernSize = 7
        
    model = convlstm(lags, features, cities, filters, kernSize)
    return model


def conv_plus_lstm(lags, features, cities, filters, kernSize):#model1
    conv1_filters = 80
    conv2_filters = 40
    conv3_filters = 1
    lstm1_nodes = 100
    lstm2_nodes = 100
    dense1_nodes = 100
    number_cities = 6
    
    input1 = Input(shape = (features, cities,lags))
    block1 = Conv2D(conv1_filters, (kernSize, kernSize), padding = 'same', activation='relu')(input1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(conv2_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Conv2D(conv3_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
    block1 = BatchNormalization()(block1)
    block1 = tensorflow.squeeze(block1,axis=-1)
    block2 = LSTM(lstm1_nodes, return_sequences=True,name = "lstm1")(block1)
    block2 = LSTM(lstm2_nodes, return_sequences=False,name = "lstm2")(block2)
    block3 = Dense(dense1_nodes, activation='relu')(block2)
    output1 = Dense(number_cities, activation='linear')(block3)
    return Model(inputs=input1, outputs=output1)

def get_conv_plus_lstm():#get_new_model1
    lags = 10
    features = 18 
    cities = 18 
    filters = 5
    kernSize = 7
    model = conv_plus_lstm(lags, features, cities, filters, kernSize)
    return model

class MSConvPlusLSTM:
    def __init__(self, window_size,conv1_filters,conv2_filters,conv3_filters,
                 conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                 padding1,padding2,padding3,conv1_activation,conv2_activation,
                 conv3_activation,dense_nodes,lstm_nodes,depth):
        
        self.number_target_cities = 6
        self.number_features = 18
        self.number_cities = 18
        
        self.window_size = window_size        
        
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        
        self.conv1_kernel_shape = conv1_kernel_shape
        self.conv2_kernel_shape = conv2_kernel_shape
        self.conv3_kernel_shape = conv3_kernel_shape
        
        self.padding1 = padding1
        self.padding2 = padding2
        self.padding3 = padding3
        
        self.conv1_activation = conv1_activation
        self.conv2_activation = conv2_activation
        self.conv3_activation = conv3_activation
        
        self.dense_nodes = dense_nodes
        
        self.lstm_nodes = lstm_nodes
        
        self.depth = depth
        
        self.model = self.get_model()

    def get_model(self):
      inputs = []
      convs = []
      for i in range(self.window_size):
          input_layer = Input(shape=(self.number_features, self.number_cities, self.depth), name = "input"+str(i+1))
          inputs.append(input_layer)

      for i in range(self.window_size):
          conv1 = Conv2D(self.conv1_filters, self.conv1_kernel_shape, padding = self.padding1, activation=self.conv1_activation, name = str(i+1)+"conv"+str(1))(inputs[i])          
          conv2 = Conv2D(self.conv2_filters, self.conv2_kernel_shape, padding = self.padding2, activation=self.conv1_activation,name = str(i+1)+"conv"+str(2))(conv1)          
          conv3 = Conv2D(self.conv3_filters, self.conv3_kernel_shape, padding = self.padding3, activation=self.conv1_activation,name = str(i+1)+"conv"+str(3))(conv2)
          conv4 = Conv2D(1,(1,1),padding="valid")(conv3)
          flat = Flatten(name = str(i+1)+"flatten")(conv4)
          lambda_layer = Lambda(lambda X: tensorflow.expand_dims(X, axis=1))(flat)
          convs.append(lambda_layer)
      
      merge = concatenate(convs,axis=1,name = "merge")  
      lstm1 = LSTM(self.lstm_nodes, return_sequences=False,name = "lstm")(merge)
      dense = Dense(self.dense_nodes, activation="sigmoid",name = "dense2")(lstm1)
      output = Dense(self.number_target_cities,activation="linear",name = "dense3")(dense)
      
      model = Model(inputs=inputs, outputs=output)
      return model  


def get_ms_conv_plus_lstm_model():#get_cascade_model
    window_size = 10
    
    conv1_filters = 1
    conv2_filters = 2
    conv3_filters = 4
    
    conv1_kernel_shape = (7,7)
    conv2_kernel_shape = conv1_kernel_shape
    conv3_kernel_shape = conv1_kernel_shape
    
    padding1 = "same"
    padding2 = padding1
    padding3 = padding1
    
    conv1_activation = "relu"
    conv2_activation = conv1_activation
    conv3_activation = conv1_activation
    
    dense_nodes = 100
    
    lstm_nodes = 175
    
    depth = 1
    cascade_object = MSConvPlusLSTM(window_size,conv1_filters,conv2_filters,conv3_filters,
                conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                padding1,padding2,padding3,conv1_activation,conv2_activation,
                conv3_activation,dense_nodes,lstm_nodes,depth)
    cascade_model = cascade_object.model
    return cascade_model


def get_ms_convlstm():#get_new_model4
    inputs = []
    convs = []
    kernel_size = 7
    window_size = 5
    lags_per_input = 2
    number_features = 18
    number_cities = 18
    number_target_cities = 6
    filters_convlstm1 = 4
    filters_convlstm2 = 4
    dense_nodes = 50
    
    for i in range(window_size):
        input_layer = Input(shape=(lags_per_input, number_features, number_cities, 1), name = "input"+str(i+1))
        inputs.append(input_layer)
    
    for i in range(window_size):
        block1 = ConvLSTM2D(filters_convlstm1, (kernel_size, kernel_size), padding = 'same', return_sequences=True, activation='relu',data_format="channels_last")(inputs[i])
        block1 = BatchNormalization()(block1)
        block1 = ConvLSTM2D(filters_convlstm2, (kernel_size, kernel_size), padding = 'same', return_sequences=False, activation='relu',data_format="channels_last")(block1)
        block1 = BatchNormalization()(block1)
        convs.append(block1)
    
    merge = concatenate(convs,axis=-1,name = "merge") 
    flat = Flatten()(merge)
    flat2 = BatchNormalization()(flat)
    dense3 = Dense(dense_nodes, activation="relu",name = "dense2")(flat2)
    output = Dense(number_target_cities,activation="linear",name = "dense3")(dense3)    
    model = Model(inputs=inputs, outputs=output)
    
    return model


from tensorflow.keras.layers import concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization




def att_unistream(lags, features, cities, filters, kernSize):#tensorial model
    dense1_nodes = 20
    dense2_nodes = 25
    number_target_cities = 6
    input1 = Input(shape = (lags, features, cities, 1))
    block1 = ConvLSTM2D(filters, (kernSize, kernSize), padding = 'same', return_sequences=True, activation='relu',data_format="channels_last")(input1)
    
    block1 = Reshape((324,50))(block1)
    
    num_heads = 1
    encoder = Encoder(num_layers=1, d_model=50, num_heads=num_heads, 
                          dff=64, input_vocab_size=8500,
                          maximum_position_encoding=100)
    block1=encoder(block1)
    
    
    
    block1 = BatchNormalization()(block1)
    block2 = Flatten(name = 'flatten')(block1)
    block3 = Dense(dense1_nodes, activation='relu')(block2)
    block3 = Dense(dense2_nodes, activation='relu')(block3)

    output1 = Dense(number_target_cities, activation='linear')(block3)
    return Model(inputs=input1, outputs=output1)

def get_att_unistream_model():#get_conv_lstm_model
    lags = 10
    features = 18 
    cities = 18 
    filters = 5
    kernSize = 7
        
    model = att_unistream(lags, features, cities, filters, kernSize)
    return model


def att_multistream():#get_new_model4
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
    reshaped_layer = Reshape((324,20))(merge)
    num_heads = 1
    encoder = Encoder(num_layers=1, d_model=20, num_heads=num_heads, 
                          dff=64, input_vocab_size=8500,
                          maximum_position_encoding=100)
    enc_layer=encoder(reshaped_layer)
    
    
    flat = Flatten()(enc_layer)
    flat2 = BatchNormalization()(flat)
    
    dense3 = Dense(dense_nodes, activation="linear",name = "dense2")(flat2)
    
    
    output = Dense(number_target_cities,activation="linear",name = "dense3")(dense3)    
    model = Model(inputs=inputs, outputs=output)
    
    return model


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  
    
    ffn_output = self.ffn(out1)  
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2,attention_weights

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
        
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training):
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    att_weights_dict = {}
    att_weights_list = []
    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
       x,attention_weights = self.enc_layers[i](x, training, None)
       att_weights_dict['encoder_layer{}'.format(i+1)] = attention_weights
       att_weights_list.append(attention_weights)
    
    return x






# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:

import numpy as np
import librosa
import glob
import os
import string
import itertools
import threading
import codecs
import unicodedata


# In[3]:

tf.logging.set_verbosity(tf.logging.DEBUG)


# In[ ]:




# In[4]:

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def text_to_char_array(original):
    r"""
    Given a Python string ``original``, remove unsupported characters, map characters
    to integers and return a numpy array representing the processed string.
    """
    # Create list of sentence's words w/spaces replaced by ''
    result = ' '.join(original.translate(None, string.punctuation).lower().split())
    result = result.replace(" '", "") # TODO: Deal with this properly
    result = result.replace("'", "")    # TODO: Deal with this properly
    result = result.replace(' ', '  ')
    result = result + ' ' #Append spaces at the end of files
    result = result.split(' ')

    # Tokenize words into letters adding in SPACE_TOKEN where required
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])
    
    # Map characters into indicies
    result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])
    
    # Add result to results
    return result




def _get_audio_feature_mfcc(wav_file):
    #All wav files are with 8k sampling rate : Taking Fourier representation: 20 ms speech to 20 feature
    sample_rate = 8000
    # load wave file with sampling rate 8000 which is already known. sr value is important
    data, sr = librosa.load(wav_file, mono=True, sr=sample_rate)

    #Short First Fourier transform - for every 20 second for 8k sampling rate= 160
    mfcc = librosa.feature.mfcc(data, sr=sample_rate, n_mfcc=20)
    
    return mfcc



def _load_feature_and_label(src_list):
    txt_file, wav_file = src_list 
    label = ''

    with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
        label = unicodedata.normalize("NFKD", open_txt_file.read()).encode("ascii", "ignore")
        label = text_to_char_array(label)
    label_len = len(label)

    feature = _get_audio_feature_mfcc(wav_file)
    feature_len = np.size(feature, 1)

    # return result
    return label, label_len, feature, feature_len





# In[ ]:




# In[5]:

###  ctc_label_dense_to_sparse and Taken from https://github.com/mozilla/DeepSpeech  ##########

# gather_nd is taken from https://github.com/tensorflow/tensorflow/issues/206#issuecomment-229678962
# 
# Unfortunately we can't just use tf.gather_nd because it does not have gradients
# implemented yet, so we need this workaround.
#
def gather_nd(params, indices, shape):
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + range(0, rank - 1)))
    flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)

# ctc_label_dense_to_sparse is taken from https://github.com/tensorflow/tensorflow/issues/1742#issuecomment-205291527
#
# The CTC implementation in TensorFlow needs labels in a sparse representation,
# but sparse data and queues don't mix well, so we store padded tensors in the
# queue and convert to a sparse representation after dequeuing a batch.

def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    # The second dimension of labels must be equal to the longest label length in the batch
    correct_shape_assert = tf.assert_equal(tf.shape(labels)[1], tf.reduce_max(label_lengths))
    with tf.control_dependencies([correct_shape_assert]):
        labels = tf.identity(labels)

    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])
    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    init = tf.expand_dims(init, 0)
    dense_mask = tf.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns),
          label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns), tf.reverse(label_shape, [0])))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)

    indices = tf.transpose(tf.reshape(tf.concat([batch_ind, label_ind], 0), [2, -1]))
    shape = [batch_size, tf.reduce_max(label_lengths)]
    vals_sparse = gather_nd(labels, indices, shape)
    
    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))


# In[6]:


#Input tensor sequence length calculation needed because of convolution changes the length
#Take input as shape`[max_time, batch_size, feature]
def get_seq_length(input_tensor):
    n_items  = tf.slice(tf.shape(input_tensor), [1], [1])
    n_steps = tf.slice(tf.shape(input_tensor), [0], [1])
    seq_length = tf.tile(n_steps, n_items)
    return seq_length


#input_tensor is of shape `[max_time, batch_size, input_size]`.
#Returns tensor of shape `[max_time, batch_size, input_size]`.
def rnn_layer(input_tensor, n_cell_units, dropout, seq_length, batch_size):

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(2*n_cell_units, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=dropout, output_keep_prob=dropout)
    
    outputs, output_states = tf.nn.dynamic_rnn(cell=lstm_fw_cell, 
                                               inputs=input_tensor,
                                               dtype=tf.float32, 
                                               time_major=True, 
                                               sequence_length=seq_length)
    
    return outputs


# In[7]:


#
# hyper parameters
#
features_in_step = 20
# The number of characters in the target language plus one ===>   (<space> + a-z + <one extra>)
n_class = 28

num_batch = 1
num_epoch = 500
batch_size = 1

####Learning Parameters
initial_learning_rate = 0.001
momentum = 0.9

#RNN Layer
n_cell_units = 128
dropout = 0.8




# In[8]:


input_labels = tf.placeholder(tf.int32, shape=[batch_size,None])
input_labels_length = tf.placeholder(tf.int32, shape=[batch_size])
input_features = tf.placeholder(tf.float32, shape=[batch_size, features_in_step, None])


weights = {
    'wr1': tf.Variable(tf.random_normal([2*n_cell_units, n_class], mean=0.1, stddev=1.0))
}

biases = {
    'br1': tf.Variable(tf.random_normal([n_class], mean=0.1, stddev=1.0))
}

conv_out = input_features

#Convert the output to time major of shape`[max_time, batch_size, feature] for RNN layer input
conv_out = tf.transpose(conv_out, perm=[2, 0, 1])

seq_length = get_seq_length(conv_out)

# RNN Layers

rnn_output = rnn_layer(conv_out, n_cell_units, dropout, seq_length, batch_size)

#Batch size x max_length x n_cell_units.
####  This is dense layer for classification -- RNN 
#ctc network performs softmax layer. in your code, rnn layer is connected to ctc loss layer. 
#output of rnn layer is internally activated, 
#so need to add one more hidden layer(as output layer) without activation function, 
#then add ctc loss layer.
prediction = tf.reshape(rnn_output, [-1, 2*n_cell_units])
prediction = tf.add(tf.matmul(prediction, weights['wr1']), biases['br1'])
prediction = tf.reshape(prediction, [batch_size, -1, n_class])
prediction = tf.transpose(prediction, perm=[1, 0, 2])


#CTC Layer
#Dense to sparse vector conversion
sparse_labels = ctc_label_dense_to_sparse(input_labels, input_labels_length, batch_size)

#Train
loss_from_ctc = tf.nn.ctc_loss(inputs=prediction, labels=sparse_labels, sequence_length=seq_length, time_major=True)
loss = tf.reduce_mean(loss_from_ctc)
optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(loss)


#Accuracy Check
decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=prediction, sequence_length=seq_length, merge_repeated=False)
accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_labels))




# In[ ]:




# In[9]:

np_input_labels, np_input_labels_length, np_input_features, np_input_features_length = _load_feature_and_label(['label.txt','audio.wav'])

np_input_labels = np.reshape(np_input_labels, (batch_size, -1))
np_input_labels_length = np.reshape(np_input_labels_length, (batch_size))
np_input_features = np.reshape(np_input_features, (batch_size, features_in_step, -1))
np_input_features_length = np.reshape(np_input_features_length, (batch_size))



# In[10]:



with tf.Session() as sess:

    # initialize the variables
    sess.run(tf.global_variables_initializer())

    print('Training')
    try:
        
        feed = {input_labels: np_input_labels, input_labels_length: np_input_labels_length, input_features: np_input_features}
    
        for epoch in xrange(num_epoch):
            epoch_loss = 0
            for step in xrange(num_batch):
                print (sess.run([loss, accuracy], feed_dict=feed))
                _, c = sess.run([optimizer, loss], feed)
                print 'Epoch:', epoch ,'Step:', step, ' Loss:', c
                epoch_loss += c
            print 'Epoch:', epoch, ' AccLoss:', epoch_loss
            
        print 'Finished.'
    except Exception, e:
        print ('Exception in code.')
    finally:
        sess.close()



# In[ ]:




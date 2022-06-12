import keras.backend as K
from keras.layers import Multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))  # 标准正态分布随机特征值
    y = np.random.randint(low=0, high=2, size=(n, 1))  # 二分类，随机标签值
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))  # 将第attention_column个column的值置为标签值
    return x, y


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    #    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    #        if print_shape_only:
    #            print(layer_activations.shape)
    #        else:
    #            print(layer_activations)
    return activations


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_after_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    K.clear_session()  # 清除之前的模型，省得压满内存
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = True
INPUT_DIM = 2
TIME_STEPS = 20

if __name__ == '__main__':

    np.random.seed(1337)  # for reproducibility

    # if True, the attention vector is shared across the input_dimensions where the attention is applied.

    N = 300000
    # N = 300 -> too few = no training
    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    #    for i in range(0,3):
    #        print(inputs_1[i])
    #        print(outputs[i])
    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        #        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()


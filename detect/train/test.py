import h5py

f = h5py.File('./saved_model/traincontrastv3_150batch_size32KLSTM64epoch100_train0.6_2_dimension_data3_5_8_120230511-103352.h5', 'r')
print(f.attrs.get('keras_version'))
import keras
print(keras.__version__)

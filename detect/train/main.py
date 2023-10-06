import io
import itertools
from datetime import datetime

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Conv1D, MaxPooling1D, CuDNNLSTM
from attention import Attention
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Bidirectional, ReLU
from keras import initializers, regularizers, constraints
import keras
# Input data files are available in the "./input_data/" directory.
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from keras import backend as k

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

epochs = 50
emb_dim = 128
batch_size = 8


# LSTM Model
# tune Dropout(0.6) and LSTM 64 or 32

# ==========================================
# ==========================================

# ============== Defining functions ==============
def label(df):
    # label data
    df.loc[df['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
    df.loc[df['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1


def preprocess(df):
    n_most_common_words = 1000  # 8000
    max_len = 20480

    # Class Tokenizer - This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary)
    # tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer = Tokenizer(num_words=n_most_common_words, lower=False)

    # fit_on_texts - Updates internal vocabulary based on a list of texts. In the case where texts contains lists, we assume each entry of the lists to be a token.
    # tokenizer.fit_on_texts(increased_vul['OPCODE'].values)
    tokenizer.fit_on_texts(df['OPCODE'].values)

    # # Transforms each text in texts in a sequence of integers.
    sequences = tokenizer.texts_to_sequences(df['OPCODE'].values)
    # sequences = tokenizer.texts_to_sequences(tt)

    # Find number of unique words/tokens
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # pad sequences with zeros in front to make them all maxlen
    X = pad_sequences(sequences, maxlen=max_len)
    # print(X)
    return X


def dftoXY(df):
    # Save test X and y
    X_test = preprocess(df)
    # label data
    label(df)
    #     df['LABEL'] = 0
    #     df.loc[df['LABEL'] == '1 0 0 0', 'LABEL'] = 0
    #     df.loc[df['LABEL'] != '1 0 0 0', 'LABEL'] = 1
    # print(pd.value_counts(df['LABEL']))
    y_test = to_categorical(df['LABEL'], num_classes=2)
    return X_test, y_test


def XandY(posdf, negdf):
    dfset = pd.concat([posdf, negdf])
    dfset = dfset.sample(frac=1, random_state=25, replace=False)

    # One-hot encode the lab
    dfset.loc[dfset['CATEGORY'] == '1 0 0 0', 'LABEL'] = 0
    dfset.loc[dfset['CATEGORY'] != '1 0 0 0', 'LABEL'] = 1
    # df_train.head()

    X, y = dftoXY(dfset)

    print('Shape of X: {}'.format(X.shape))

    # for sm.fit_sample
    y_labels = np.expand_dims(np.array(np.argmax(y, axis=1)), axis=1)
    print('Shape of y: {}'.format(y_labels.shape))
    return X, y_labels


# ============== Loading and reading csv input data ==============
dataset = 'test_data_3.csv'
data = pd.read_csv('./dataset/' + dataset, usecols=['ADDRESS', 'OPCODE', 'CATEGORY'])




# ========== set of vul contracts ==========
# shuffle positives dataset
X_test = preprocess(data)
# ============ Resample ============




from keras.models import load_model

# 加载模型
time1 = datetime.now()
model = load_model('./saved_model/traincontrastv3_150batch_size32KLSTM64epoch100_train0.6_2_dimension_data3_5_8_120230521-203841.h5', custom_objects={'Attention': Attention})
#model = load_model('your_model_path.h5', custom_objects={'AttentionLayer': attention.AttentionLayer})
y_pred = model.predict(X_test, batch_size=32, verbose=0)
time2 = datetime.now()
print(time2 - time1)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

import sklearn

class_names = ["negative", "positive"]
# Calculate the confusion matrix.y
cm = sklearn.metrics.confusion_matrix(ytest_true, y_pred)
# Log the confusion matrix as an image summary.
figure = plot_confusion_matrix(cm, class_names=class_names)
cm_image = plot_to_image(figure)

from sklearn.metrics import average_precision_score
# Compute the average precision score
average_precision = average_precision_score(ytest_true, y_pred)
print('Average Precision Score: {:0.4f}\n'.format(average_precision))

# Compute the recall
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(ytest_true, y_pred)
print('Recall Score: {:0.4f}\n'.format(recall[1]))

print("============== ADDED Evaluation ==============")
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from sklearn.metrics import average_precision_score


# Compute the average precision score
average_precision = average_precision_score(ytest_true, y_pred)
print('Average Precision Score: {:0.4f}\n'.format(average_precision))

# Compute the recall
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(ytest_true, y_pred)
print('Recall Score: {:0.4f}\n'.format(recall[1]))

print("============== ADDED Evaluation ==============")
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

# The set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
print('Accuracy: {:0.4f}\n'.format(accuracy_score(ytest_true, y_pred)))
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
print('Recall: {:0.4f}\n'.format(recall_score(ytest_true, y_pred)))
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
print('Precision: {:0.4f}\n'.format(precision_score(ytest_true, y_pred)))
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
print('F1 score: {:0.4f}\n'.format(f1_score(ytest_true, y_pred)))

# In binary classification, the count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}.
print('\n confusion matrix:\n', confusion_matrix(ytest_true, y_pred))
# Overview of all scores
print('\n clasification report:\n', classification_report(ytest_true, y_pred))

# ============== Save Model and Results ==============







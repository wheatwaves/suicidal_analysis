from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import cPickle
from keras.utils import np_utils
X = [[[1,1],[1,1]],[[2,2],[2,2]]]
Y=[[0,0],[1,1]]
y = np_utils.to_categorical(Y)
model = Sequential()
model.add(LSTM(output_dim=2, return_sequences = True, activation='sigmoid', inner_activation='hard_sigmoid', input_shape = (2,2)))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['binary_accuracy','recall','precision'])
model.fit(X, Y, batch_size = len(X), nb_epoch=10)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import cPickle

X=[[1,2,3],[4,5,6]]
Y=[0,1]
model = Sequential()
model.add(LSTM(output_dim=2, return_sequences = True, activation='sigmoid', inner_activation='hard_sigmoid', input_shape = (1,3)))
# model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['binary_accuracy','recall','precision'])
model.fit(X, Y, batch_size = 1, nb_epoch=10)

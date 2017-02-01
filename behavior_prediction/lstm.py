from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import cPickle
SU = ['CC0001', 'CC0005', 'CC0006', 'CC0008', 'CC0011', 'CC0013']
CO = ['CC0004', 'CC0020', 'CC0021', 'CC0022', 'CC0050', 'CC0051']
MH = ['CC0002', 'CC0010', 'CC0028', 'CC0031', 'CC0035', 'CC0036']
batch_size = 32
DATA_DIR = '../data/svm_input/'
history = 2.0 #measured in seconds
total_time = 120.0
feature_size = 14
output_dim = 64
def construct_sequence_data(filename):
	data = cPickle.load(filename)
	X,Y,sequence_X,sequence_Y = [],[],[],[]
	history_len = int(len(data)*history/total_time)
	for line in data:
		X.append(line[-1-feature_size:-1])
		Y.append(line[-1])
	for i in xrange(history_len,len(X)):
		sequence_X.append(X[i-history_len:i+1])
		sequence_Y.append(Y[i])
	return sequence_X, sequence_Y
def load_data(train_file,validation_file,test_file):
	train_X, train_Y, validation_X, validation_Y, test_X, test_Y = [],[],[],[],[],[]
	for filename in train_file:
		X,Y = construct_sequence_data(open(DATA_DIR+filename))
		for x in X:
			train_X.append(x)
		for y in Y:
			train_Y.append(y)
	for filename in validation_file:
		X,Y = construct_sequence_data(open(DATA_DIR+filename))
		for x in X:
			validation_X.append(x)
		for y in Y:
			validation_Y.append(y)
	for filename in test_file:
		X,Y = construct_sequence_data(open(DATA_DIR+filename))
		for x in X:
			test_X.append(x)
		for y in Y:
			test_Y.append(y)
	return train_X, train_Y, validation_X, validation_Y, test_X, test_Y



def train_lstm(train_X, train_Y, validation_X, validation_Y):
	model = Sequential()
	model.add(LSTM(output_dim=output_dim, activation='sigmoid', inner_activation='hard_sigmoid', dropout_W=0.2, dropout_U=0.2))
	# model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch=10, validation_data=(validation_X, validation_Y))
	score = model.evaluate(X_test, Y_test, batch_size = batch_size)


if __name__ == '__main__':
	train_file = SU[:1]+CO[:1]+MH[:1]
	validation_file = SU[3]+CO[3]+MH[3]
	test_file = SU[4:]+CO[4:]+MH[4:]
	train_X, train_Y, validation_X, validation_Y, test_X, test_Y = load_data(train_file,validation_file,test_file)
	train_lstm(train_X, train_Y, validation_X, validation_Y)



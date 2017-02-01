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
histories = [5,10,15,20]
feature_size = 14
output_dims = [16,32,64,128]
dropouts = [0.1,0.3,0.5]
def construct_sequence_data(filename,history):
	data = cPickle.load(filename)
	X,Y,sequence_X,sequence_Y = [],[],[],[]
	for line in data:
		X.append(line[-1-feature_size:-1])
		Y.append([line[-1]])
	for i in xrange(history,len(X)):
		sequence_X.append(X[i-history:i+1])
		sequence_Y.append(Y[i])
	return sequence_X, sequence_Y
def load_data(train_file,validation_file,test_file,history):
	train_X, train_Y, validation_X, validation_Y, test_X, test_Y = [],[],[],[],[],[]
	for filename in train_file:
		X,Y = construct_sequence_data(open(DATA_DIR+filename),history)
		for x in X:
			train_X.append(x)
		for y in Y:
			train_Y.append(y)
	for filename in validation_file:
		X,Y = construct_sequence_data(open(DATA_DIR+filename),history)
		for x in X:
			validation_X.append(x)
		for y in Y:
			validation_Y.append(y)
	for filename in test_file:
		X,Y = construct_sequence_data(open(DATA_DIR+filename),history)
		for x in X:
			test_X.append(x)
		for y in Y:
			test_Y.append(y)
	return train_X, train_Y, validation_X, validation_Y, test_X, test_Y



def train_lstm(train_X, train_Y, validation_X, validation_Y, output_dim, history, dropout):
	model = Sequential()
	model.add(LSTM(output_dim=output_dim, activation='sigmoid', inner_activation='hard_sigmoid', dropout_W=dropout, dropout_U=dropout, input_shape = (history+1,feature_size)))
	# model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['binary_accuracy','recall','precision'])

	model.fit(train_X, train_Y, batch_size = batch_size, nb_epoch=10, validation_data=(validation_X, validation_Y))
	# score = model.evaluate(X_test, Y_test, batch_size = batch_size)

def grid_search(train_file, validation_file, test_file):
	for history in histories:
		for output_dim in output_dims:
			for dropout in dropouts:
				train_X, train_Y, validation_X, validation_Y, test_X, test_Y = load_data(train_file,validation_file,test_file,history)
				train_lstm(train_X, train_Y, validation_X, validation_Y, output_dim, history, dropout)
if __name__ == '__main__':
	train_file = SU[:4]+CO[:4]+MH[:4]
	validation_file = SU[4:]+CO[4:]+MH[4:]
	test_file = SU[4:]+CO[4:]+MH[4:]
	grid_search(train_file, validation_file, test_file)



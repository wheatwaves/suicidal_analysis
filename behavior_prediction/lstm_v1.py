from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import cPickle
SU = ['CC0001', 'CC0005', 'CC0006', 'CC0008', 'CC0011', 'CC0013']
CO = ['CC0004', 'CC0020', 'CC0021', 'CC0022', 'CC0050', 'CC0051']
MH = ['CC0002', 'CC0010', 'CC0028', 'CC0031', 'CC0035', 'CC0036']
DATA_DIR = '../data/svm_input_with_framerate_30/'
feature_size = 14
output_dims = [64]
dropouts = [0.5]
def load_data(train_file,validation_file,test_file):
	train_X, train_Y, validation_X, validation_Y, test_X, test_Y = [],[],[],[],[],[]
	for filename in train_file:
		data = cPickle.load(open(DATA_DIR+filename))
		file_X, file_Y = [],[]
		for line in data:
			file_X.append(line[-1-feature_size:-1])
			file_Y.append(line[-1:0])
		train_X.append(file_X)
		train_Y.append(file_Y)
	for filename in validation_file:
		data = cPickle.load(open(DATA_DIR+filename))
		file_X, file_Y = [],[]
		for line in data:
			file_X.append(line[-1-feature_size:-1])
			file_Y.append([line[-1:0]])
		validation_X.append(file_X)
		validation_Y.append(file_Y)
	for filename in test_file:
		data = cPickle.load(open(DATA_DIR+filename))
		file_X, file_Y = [],[]
		for line in data:
			file_X.append(line[-1-feature_size:-1])
			file_Y.append(line[-1:0])
		test_X.append(file_X)
		test_Y.append(file_Y)
	return train_X, train_Y, validation_X, validation_Y, test_X, test_Y



def train_lstm(train_X, train_Y, validation_X, validation_Y, output_dim, dropout):
	model = Sequential()
	model.add(LSTM(output_dim=output_dim, return_sequences = True, activation='sigmoid', inner_activation='hard_sigmoid', dropout_W=dropout, dropout_U=dropout, input_shape = (len(train_X[0]),len(train_X[0][0])) ))
	# model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy',
	              optimizer=adam,
	              metrics=['binary_accuracy','recall','precision'])
	
	model.fit(train_X, train_Y, batch_size = len(train_X), nb_epoch=10, validation_data=(validation_X, validation_Y))
	# score = model.evaluate(X_test, Y_test, batch_size = batch_size)

def grid_search(train_file, validation_file, test_file):
	for output_dim in output_dims:
		for dropout in dropouts:
			train_X, train_Y, validation_X, validation_Y, test_X, test_Y = load_data(train_file,validation_file,test_file)
			train_lstm(train_X, train_Y, validation_X, validation_Y, output_dim, history, dropout)
if __name__ == '__main__':
	train_file = SU[:4]+CO[:4]+MH[:4]
	validation_file = SU[4:]+CO[4:]+MH[4:]
	test_file = SU[4:]+CO[4:]+MH[4:]
	grid_search(train_file, validation_file, test_file)



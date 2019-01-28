''' The training Module to train the SpatioTemporal AutoEncoder

Run:

>>python3 train.py n_epochs(enter integer)     to begin training.





Author: Harsh Tiku


'''

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from model import load_model
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_files(path):
	filelist = glob.glob(os.path.join(path, "*"))
	for f in filelist:
		os.remove(f)

parser=argparse.ArgumentParser()
parser.add_argument('n_epochs',type=int)
parser.add_argument('log_dir',type=str)

args=parser.parse_args()

X_train=np.load('training.npy')
frames=X_train.shape[2]
#Need to make number of frames divisible by 10


frames=frames-frames%10

X_train=X_train[:,:,:frames]
X_train=X_train.reshape(-1,227,227,10)
X_train=np.expand_dims(X_train,axis=4)
Y_train=X_train.copy()


log = args.log_dir
epochs=args.n_epochs
batch_size=1


#Make a temp dir to store all the frames
create_dir(log)

#Remove old images
remove_old_files(log)

if __name__=="__main__":

	model=load_model()

	callback_save = ModelCheckpoint("AnomalyDetector.h5",
									monitor='val_acc', save_best_only=True)

	callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

	callback_tb = TensorBoard(log_dir=log, histogram_freq=0,
							  write_images=True)

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	print('Model has been loaded and saved')

	history = model.fit(X_train,Y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  shuffle=False,
			  validation_split=0.3,
			  verbose=1,
			  callbacks = [callback_save,callback_early_stopping,callback_tb]
			  ).history

	plt.figure(figsize=(12, 8))
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.ylabel('Loss', fontsize=18)
	plt.xlabel('Epoch', fontsize=18)
	plt.legend(['train', 'test'], loc='upper right', fontsize=18)
	plt.savefig(log + '/train_loss.png')
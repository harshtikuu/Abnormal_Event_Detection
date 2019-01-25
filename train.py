''' The training Module to train the SpatioTemporal AutoEncoder

Run:

>>python3 train.py n_epochs(enter integer)     to begin training.





Author: Harsh Tiku


'''

from keras.callbacks import ModelCheckpoint, EarlyStopping
from model import load_model
import numpy as np 
import argparse



parser=argparse.ArgumentParser()
parser.add_argument('n_epochs',type=int)

args=parser.parse_args()

X_train=np.load('training.npy')
frames=X_train.shape[2]
#Need to make number of frames divisible by 10


frames=frames-frames%10

X_train=X_train[:,:,:frames]
X_train=X_train.reshape(-1,227,227,10)
X_train=np.expand_dims(X_train,axis=4)
Y_train=X_train.copy()



epochs=args.n_epochs
batch_size=1



if __name__=="__main__":

	model=load_model()

	callback_save = ModelCheckpoint("AnomalyDetector.h5",
									monitor="mean_squared_error", save_best_only=True)

	callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

	print('Model has been loaded')

	model.fit(X_train,Y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  callbacks = [callback_save,callback_early_stopping]
			  )
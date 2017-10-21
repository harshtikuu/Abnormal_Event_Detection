from model import load_model
import processor
import numpy as np 

X_train=np.load('imagestore.npy')
X_train=np.expand_dims(X_train,axis=1)
Y_train=X_train.copy()



epochs=20
batch_size=20



if __name__=="__main__":

	model=load_model()


	print('Model has been loaded')

	model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs)
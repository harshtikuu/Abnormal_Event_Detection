''' Source Code for the SpatioTemporal AutoEncoder as described in the paper

Abnormal Event Detection in Videos using Spatiotemporal Autoencoder
by Yong Shean Chong Yong Haur Tay
Lee Kong Chian Faculty of Engineering Science, Universiti Tunku Abdul Rahman, 43000 Kajang, Malaysia.

Implemented in keras



The model has over a Million trainable Params so I recommend training it on a GPU.



The model takes input a batch of 10 of Video frames of size (227,227) (grayscaled)

Extracts spatial and temporal Information and computes the reconstruction loss by Euclidean Distance b/w
original batch and Reconstructed batch



See model summary as:

>>from model import load_model
>>mod=load_model()
>>mod.summary()





Author: Harsh Tiku

'''



from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential



def load_model():
	"""
	Return the model used for abnormal event 
	detection in videos using spatiotemporal autoencoder

	"""
	model=Sequential()
	model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
	model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))



	model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))

	
	model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))


	model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))




	model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
	model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

	model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

	return model



from keras.preprocessing.image import img_to_array,load_img
from sklearn.preprocessing import StandardScaler
import numpy as np 
import os 
from scipy.misc import imresize 



imagestore=[]


pwd=os.getcwd()

def store(image):

	global imagestore



	imagepath=os.path.realpath(image)
	img=load_img(imagepath)
	img=img_to_array(img)
	img=img-img.mean()
	img=img/img.std()
	img=imresize(img,(227,227,3))

	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]

	imagestore.append(gray)



os.chdir('./train/outputs')
images=os.listdir()

for image in images:
	store(image)


os.chdir(pwd)

imagestore=np.array(imagestore)
a,b,c=imagestore.shape
imagestore.resize(b,c,a)
np.save('imagestore.npy',imagestore)
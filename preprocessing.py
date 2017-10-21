from keras.preprocessing.image import load_img,img_to_array
from scipy.misc import imresize
import numpy as np 
import os
from sklearn.preprocessing import StandardScaler



imagestore=[]


def store(image):

	global imagestore



	imagepath=os.path.realpath(image)
	print(imagepath)
	'''img=load_img(imagepath)
	img=img_to_array(img)
	img=StandardScaler().fit(img)
	img=img-img.mean()
	img=img/img.std()
	img=imresize(img,(227,227,3))

	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]

	imagestore.append(gray)'''








os.chdir('./train')
os.mkdir('./outputs')
videos=os.listdir()

for video in videos[1:]:
	path=os.path.realpath(video)
	if os.path.isfile(path):
		os.system( 'ffmpeg -i {} -r 1/20  ./outputs/%03d.jpg'.format(path))

	images=os.listdir('./outputs')
	for image in images:
		store(image)



#imagestore=np.array(imagestore)
#np.save('imagestore.npy',imagestore)
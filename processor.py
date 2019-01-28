''' This module extracts frames a Video
performs preprocessing on the frames and stores them in a Numpy array for
furthur use by the spatiotemporal autoencoder

___________________________________________________________________

Dependencies: ffmpeg

If you dont have ffmpeg installed:

Install it with :


1. sudo apt-get install ffmpeg for Linux Users
2. brew install ffmpeg for macOS

__________________________________________________________________

Usage:

python3 processor.py video_dir_path time_in_seconds_to_extract_one_frame

eg;python3 processor.py ./train 5   will search for train directory and for each video in train directory

It will extract 1 frame every 5 seconds and store it.



__________________________________________________________


Author: Harsh Tiku
'''









from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
from scipy.misc import imresize 
import argparse

imagestore=[]


parser=argparse.ArgumentParser(description='Source Video path')
parser.add_argument('source_vid_path',type=str)
parser.add_argument('fps',type=int)
args=parser.parse_args()

video_source_path= args.source_vid_path
fps=args.fps

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.png"))
	for f in filelist:
		os.remove(f)

def store(image_path):
	img=load_img(image_path)
	img=img_to_array(img)


	#Resize the Image to (227,227,3) for the network to be able to process it. 


	img=imresize(img,(227,227,3))

	#Convert the Image to Grayscale


	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]

	imagestore.append(gray)



#List of all Videos in the Source Directory.
videos=os.listdir(video_source_path)
print("Found ",len(videos)," training video")


#Make a temp dir to store all the frames
create_dir(video_source_path+'/frames')

#Remove old images
remove_old_images(video_source_path+'/frames')

framepath=video_source_path+'/frames'

for video in videos:
		os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))
		images=os.listdir(framepath)
		for image in images:
			image_path=framepath+ '/'+ image
			store(image_path)


imagestore=np.array(imagestore)
a,b,c=imagestore.shape
#Reshape to (227,227,batch_size)
imagestore.resize(b,c,a)
#Normalize
imagestore=(imagestore-imagestore.mean())/(imagestore.std())
#Clip negative Values
imagestore=np.clip(imagestore,0,1)
np.save('training.npy',imagestore)
#Remove Buffer Directory
os.system('rm -r {}'.format(framepath))

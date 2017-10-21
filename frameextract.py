import os

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
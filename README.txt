# Face mask detection

Face mask detection system to identify unmasked people using python.

How to run the detector:

1.Download all the files from the GitHub repository " https://github.com/Tihor3393/Facemask_detector "
Make sure that you have all the files are in same directory.

2.Download datasets from this link " https://drive.google.com/drive/folders/1HgfcJ8rWpuhZIXeAOk1dyvNamqvAdLKY?usp=sharing "

3.In FaceMask_Train.ipynb update the 5 directories of the dataset and also both "with mask" and "without mask" folders separately.

4.Run code in 'FaceMask_Train.ipynb' file and then it will create 'data.npy', 'labels.npy' and 'MaskNet1.hdf5'

5.We are ready to test our Face mask detector tool before running Main.ipynb update the directory of haarcascade_frontalface_default.xml

6.Now run code 'Main.ipynb', then a window opens which captures the video stream from the camera and displays the results in the real time. Place your face in the blue circle for the detector to work. 

7.To exit the live stream popup press "q"
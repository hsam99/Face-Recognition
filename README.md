# Face-Recognition
The pre-trained deep CNN model file is available in think link, you have to download the file and place it within the same directory.
https://drive.google.com/file/d/10bebUbmsUo5QVpdbJ1B6v_XlR0H3ACpL/view?usp=sharing

The main code to be run is Evaluation.m. The code will call the baseline face recognition method (FaceRecognition.m), the face recognition method using HOG feature descriptor (FaceREcognition1.m) and the deep learning face recognition method (FaceRecognition2.m).

The train and test images are located in FaceDatabase. You can use your own face dataset by using the same file structure. Inside the Train folder, the name of the subfolder will be the label for the images within it, and different subfolders contain images of different faces. The Test folder contains testing images where you have to modify testLabel.m to match with the label of the images in this folder.

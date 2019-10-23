# comma_ai_monitoring_challenge
Submission for the Comma AI Driver Monitoring Challenge

# Dlib
Dlib's facial landmark detection model will be used in this project as the feature extractor. The pre-trained model can be found here:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### Adjusting Contrast/Brightness and Facial Detection
The first addressed problem of this challenge is that the video can be of various contrast and brightness. In order for dlib's facial landmark detection to be most effective, it was best to automatically adjust these features based on a greyscale histogram of the frame. In addition, because we know that Comma.ai currently only supports right-hand driving, and that the Snapdragon will always have the same orientation, the frames are cropped to the right in order to make facial detection easier. Once the frame is processed by adjusting the contrast/brightness and cropping to the face, the image is passed to the dlib model, which returns 68 points that correspond to various locations of a face such as nose, eyes, and eyebrows. cv2's solvePnP solves the "Perspective-n-Point" problem and allows us to extract the Euler Angles of the face that is detected by the model. Plotting of the 68 points and a 3D box can be seen in the pictures below. On the left are the contrast and brightness adjusted images:
![alt text](https://raw.githubusercontent.com/seccode/comma_ai_monitoring_challenge/master/group_1.png)

### Inattentiveness Detection
The determination of inattentiveness is done by a support vector machine that uses 4 features:
- Eyes Area
- Head Position (3 Euler Angles)

The feature vector is the past 30 frames of each of these features. The training of the SVM is done within the main script by calling the```bash
--label 1 ```flag, and holding down either the 0 or 1 key to label the current frame attentive and inattentive respectively. While definitions are not given for inattentiveness, the assumption is made that not looking ahead and having eyes closed constitutes inattentiveness.

The labeling of the videos is done and the features/labels are saved so that when the main script is run by calling the```bash
--predict 1 ```flag, all feature/label pairs are used to initialize/train the SVM and perform predictions on the given video.

The SVM's probabilistic classification of the current frame is shown by the top bar, with the percentages displayed underneath. There is a threshold of time of inattentiveness before an alert is given; this is logical that not every instance of inattentiveness will cause alarm, as mistakes or quick glances away from the road should not do this. The two states of inattentiveness are shown in the examples below:
![alt text](https://raw.githubusercontent.com/seccode/comma_ai_monitoring_challenge/master/group_2.png)

### Videos 3 and 4
Driver examples 3 and 4 posed a challenge in that both drivers were wearing glasses that significantly impaired the ability of the network to accurately distinguish their faces - especially their eyes. This challenge is mitigated by the facial detections averaging out to approximate head position and gaze direction fairly well. The support vector machine is not able to determine if the driver's eyes are open in these instances, but head position can still be used. There is a good argument that with enough data, eye-gaze/eyes being open can be well approximated by the angular position of the head.
![alt text](https://raw.githubusercontent.com/seccode/comma_ai_monitoring_challenge/master/group_3.png)

### Possible Issues
Ideally we would add the speed of the car to the feature vector. When turning, the speed of the car will be low and a driver looking out the window should not be classified as inattentive because they are looking where they are turning.


#

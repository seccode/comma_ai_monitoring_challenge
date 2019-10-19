import cv2
import dlib
import numpy as np
from imutils import face_utils
import argparse
from sklearn.svm import SVC
import glob

'''
Idea:
    Use Dlib's facial landmark detection model to extract facial features,
    and train a Support Vector Machine to Classify driving instances as either
    'Attentive' or 'Inattentive'. The feature vector for the SVM is a 30,4 matrix
    where the features are Eye:Face Area Ratio and the 3 Euler Angles of the
    Head Rotation for the past 30 frames. Every frame is classifed by the SVM,
    and a heuristic is set as 10 consecutive frames of inattentive classifications
    being the definition of inattentiveness.

Labeling:
    Labels for the SVM are either Attentive or Inattentive (0 or 1). When the
    --label flag is passed as an argument, holding the '0' and '1' will label
    the feature vector representing the current frame as attentive and inattentive
    respectively. Labeling different videos or the same video over and over will
    not overwrite the feature/label pairs generated, but will add more data to
    the .npy files in data/. When a video is called with the --predict flag, all
    the feature/label pairs in the data file will be loaded and used to train
    the SVM before predicting frames of the video.
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Pass video file and labeling/predction information")
    parser.add_argument("--video",dest="video",default="0",
                        help="Path to video file")
    parser.add_argument("--label",dest="label",default=0,
                        help="1 to label video data, labeling is done by \
                        pressing the '1' key while video is played (or in real \
                        time webcam) for inattentive; the default class is \
                        attentive if no key is pressed for a frame")
    parser.add_argument("--predict",dest="predict",default=0,
                        help="1 to train SVM on given data and predict frames")
    return parser.parse_args()


class FaceDetector:
    def __init__(self):
        self.dist_coeffs = np.zeros((4,1))
        self.face_landmark_path = 'model/shape_predictor_68_face_landmarks.dat'
        self.head_rotation = np.zeros((3,)) # (3,) Store head rotation information

    def get_model_points(self,filename='model/model.txt'):
            raw_value = []
            with open(filename) as file:
                for line in file:
                    raw_value.append(line)
            model_points = np.array(raw_value, dtype=np.float32)
            model_points = np.reshape(model_points, (3, -1)).T
            model_points[:, 2] *= -1
            return model_points

    def annotate_face(self, frame,
                                shape,
                                rotation_vector,
                                translation_vector,
                                color=(255, 255, 255),
                                line_width=2):
            # Draw 3D box on frame in front of head
            point_3d = []
            rear = 75
            front_side = 120
            front_top = 120
            front_depth = 100
            for size, depth in zip([(rear,rear),(front_side,front_top)],[0,front_depth]):
                for mults in [(-1,-1),(-1,1),(1,1),(1,-1),(-1,-1)]:
                    point_3d.append((size[0]*mults[0],size[1]*mults[1],depth))
            point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

            # Project 3D points onto 2D frame
            point_2d = np.int32(cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeffs)[0].reshape(-1, 2))

            # If head is not facing straight ahead, color head box green
            if point_2d[0][0] < point_2d[5][0] or \
                point_2d[0][1] < point_2d[5][1] or \
                point_2d[2][0] > point_2d[7][0] or \
                point_2d[2][1] > point_2d[7][1]:
                color = (0,255,0)

            # Add lines to frame
            cv2.polylines(frame, [point_2d], True, color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[1]), tuple(
                point_2d[6]), color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[2]), tuple(
                point_2d[7]), color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[3]), tuple(
                point_2d[8]), color, line_width, cv2.LINE_AA)

            # Add feature points to frame
            for i, item in enumerate(shape):
                if i >= 36 and i <= 47: # Eye feature points
                    cv2.circle(frame,tuple(item),1,(0,255,0),-1)
                else:
                    cv2.circle(frame,tuple(item),1,(255,0,0),-1)

    def get_head_pose(self,shape):
        # Find rotational and translational vectors of head position
        image_pts = np.float32([shape])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.model_points_68,
                                                        image_pts,
                                                        self.camera_matrix,
                                                        self.dist_coeffs)

        rvec_matrix = cv2.Rodrigues(rotation_vec)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vec))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        return (rotation_vec, translation_vec, eulerAngles)

    def eyes_area_per_face_area(self,shape):
        # Calculate average eye area and update running tracker
        eye_left = np.array([shape[36],shape[37],shape[38],
                            shape[39],shape[40],shape[41]])
        eye_right = np.array([shape[42],shape[43],shape[44],
                            shape[45],shape[46],shape[47]])

        face_outline = np.array([shape[1],shape[9],shape[17]])
        face_area = self.shoelace_formula(face_outline)

        left_area = self.shoelace_formula(eye_left) / face_area
        right_area = self.shoelace_formula(eye_right) / face_area
        return (left_area+right_area)/2

    def shoelace_formula(self,points):
        # Determine area of polygon from ordered points
        x = points[:,0]
        y = points[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def train_svm(self):
        # Train and return Support Vector Machine on available data
        all_features = np.array([[0,0,0,0]])
        all_labels = np.array([0])
        feature_files = glob.glob('data/features_*')
        label_files = glob.glob('data/labels_*')
        assert len(feature_files) > 0, "No saved feature vectors"
        for f_file, l_file in zip(feature_files,label_files):
            all_features = np.concatenate((all_features,np.load(f_file)))
            all_labels = np.concatenate((all_labels,np.load(l_file)))

        clf = SVC(class_weight='balanced',probability=True)
        clf.fit(all_features,all_labels)
        print('Training Error: {}'.format(clf.score(all_features,all_labels)))
        return clf

    def preprocess_frame(self,frame):
        # Crop frame
        frame = frame[:,int(frame.shape[0]/2):]

        # Resize frame for faster detection
        frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
        frame_orig = frame.copy()

        # Adjust contrast and brightness of frame
        # Get histogram for grey scale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        # Get CDF of histogram
        cdf = [float(hist[0])]
        [cdf.append(cdf[-1] + float(hist[index])) for index in range(1, len(hist))]

        hist_percent = (cdf[-1]*0.125)
        min_gray = np.argmin(np.abs(np.array(cdf) - hist_percent))
        max_gray = np.argmin(np.abs(np.array(cdf) - cdf[-1] + hist_percent))

        alpha = 255 / (max_gray - min_gray)
        beta = -min_gray * alpha
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        return frame, frame_orig #Return frame copy for original frame coloring

    def detect(self):
        FEATURES = []
        LABELS = []
        # Read from video and detect head position
        if args.video == '0':
            args.video = 0
        video = cv2.VideoCapture(args.video)
        assert video.isOpened(), "Video {} not found".format(args.video)

        # Find size of frame
        _, test_frame = video.read()
        size = test_frame.shape
        self.focal_length = size[1]
        self.camera_center = (size[1] / 2, size[0] / 2)

        # 68 facial feature points
        self.model_points_68 = self.get_model_points()

        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Initiliaze dlib face predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.face_landmark_path)

        if args.predict:
            # Train SVM instantiate classifier
            clf = self.train_svm()

        # Driver state stores classifications of attentive/inattentive for
        # past 30 frames
        self.driver_state = np.zeros((30,))

        self.count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Adjust frame to make detection easier
            frame, frame_orig = self.preprocess_frame(frame)

            # Feature vector that represents driver state
            # [Eye to Face Area Ratio, Angle 1, Angle 2, Angle 3]
            current_state = np.array([[0,0,0,0]])

            # Detect face in frame
            faces = self.detector(frame, 0)

            if len(faces) > 0: # Face detected

                # Get head feature points from first detected head
                shape = face_utils.shape_to_np(self.predictor(frame_orig, faces[0]))

                # Find position of head
                rotation_vec, translation_vec, eulerAngles = self.get_head_pose(shape)

                self.count += 1
                for x in range(3): # Euler Angles
                    self.head_rotation[x] += (eulerAngles[x] - self.head_rotation[x]) / self.count

                # Get average area of right and left eye
                eyes_area = self.eyes_area_per_face_area(shape)

                # Draw facial features
                self.annotate_face(frame_orig,shape,rotation_vec,translation_vec)

                # Create translational vector adjusted for rolling average
                controlled_translational_vec = eulerAngles.flatten() - self.head_rotation

                # Update current driver state
                current_state = np.append(np.array([eyes_area]),
                                        controlled_translational_vec).reshape(1,4)

            if args.label:
                # Add state vector to features
                FEATURES.append(current_state.flatten())

            # Use SVM to determine if inattentive
            if args.predict:
                # 1 if inattentive
                probs = clf.predict_proba(current_state.reshape(1,-1)).flatten()
                pred = np.argmax(probs)
                # Update driver state vector
                self.driver_state = np.append(np.delete(self.driver_state,0),pred)

                # Show SVM Classifier Bar
                bar_length = frame.shape[1] - 30
                red_length = int(bar_length*probs[1])
                cv2.rectangle(frame_orig,(15,10),(15+red_length,40),
                                (0,0,255),cv2.FILLED)
                cv2.rectangle(frame_orig,(15+red_length,10),(bar_length,40),
                                (0,255,0),cv2.FILLED)

                if np.min(self.driver_state[-20:]) == 1: # If last 10 frames are inattentive
                    # Put red border on frame
                    frame_orig = cv2.copyMakeBorder(frame_orig, 5, 5, 5, 5,
                                                    cv2.BORDER_CONSTANT, value=(0,0,255))

                    # Put text on frame
                    cv2.putText(frame_orig,"Inattentive: {}".format(round(100*probs[1],2)),(20,80),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                else: # Attentive
                    cv2.putText(frame_orig,"Inattentive: {}%".format(round(100*probs[1],2)),(20,80),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.putText(frame_orig,"Attentive: {}%".format(round(100*probs[0],2)),(20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

            cv2.imshow("Frame", frame_orig)
            key = cv2.waitKey(1)
            # Use cv2 waitKey to label frame
            if args.label or key == 27:
                if key == 27:
                    LABELS.append(0) #Attentive
                    break
                elif key == ord('0'):
                    LABELS.append(0) #Attentive
                elif (key == ord('1') or \
                        np.max(current_state) == 0): # Inattentive if not face detected
                    LABELS.append(1) #Inattentive
                else:
                    LABELS.append(0) #Attentive

        if args.label:
            # Save labeled instances to data file
            index = len(glob.glob('data/features_*'))
            np.save('data/features_'+str(index),FEATURES)
            np.save('data/labels_'+str(index),LABELS)


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Start detection
    detector = FaceDetector()
    detector.detect()









#

import cv2
import dlib
import numpy as np
from imutils import face_utils
import argparse
import time
from sklearn.svm import SVC
import glob
import math

class FaceDetector:
    def __init__(self):
        self.dist_coeffs = np.zeros((4,1))
        self.face_landmark_path = 'model/shape_predictor_68_face_landmarks.dat'
        self.eyes_area = 0 # Store area of eye feature
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

    def draw_annotation_box(self, frame,
                                shape,
                                rotation_vector,
                                translation_vector,
                                color=(255, 255, 255),
                                line_width=2):
            # Draw 3D box on frame in front of head
            point_3d = []
            rear_size = 75
            rear_depth = 0
            point_3d.append((-rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, -rear_size, rear_depth))
            front_size_side = 120
            front_size_top = 120
            front_depth = 100
            point_3d.append((-front_size_side, -front_size_top, front_depth))
            point_3d.append((-front_size_side, front_size_top, front_depth))
            point_3d.append((front_size_side, front_size_top, front_depth))
            point_3d.append((front_size_side, -front_size_top, front_depth))
            point_3d.append((-front_size_side, -front_size_top, front_depth))
            point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

            # Project 3D points onto 2D frame
            (point_2d, _) = cv2.projectPoints(point_3d,
                                              rotation_vector,
                                              translation_vector,
                                              self.camera_matrix,
                                              self.dist_coeffs)

            point_2d = np.int32(point_2d.reshape(-1, 2))

            head_turned = False
            # If head is turned, color annotation box green
            if point_2d[0][0] < point_2d[5][0] or \
                point_2d[0][1] < point_2d[5][1] or \
                point_2d[2][0] > point_2d[7][0] or \
                point_2d[2][1] > point_2d[7][1]:
                color = (0,255,0)
                head_turned = True

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
            return head_turned

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
        # pitch_roll_yaw = [math.radians(_) for _ in eulerAngles]

        return (rotation_vec, translation_vec, eulerAngles)

    def eyes_area_per_face_area(self,shape):
        # Heuristic to determine if eyes are open based on a continuously
        # calculated average eye area controlled for head size (proxy for depth)
        eye_left = np.array([shape[36],shape[37],shape[38],
                            shape[39],shape[40],shape[41]])
        eye_right = np.array([shape[42],shape[43],shape[44],
                            shape[45],shape[46],shape[47]])

        face_outline = np.array([shape[1],shape[9],shape[17]])
        face_area = self.shoelace_formula(face_outline)

        left_area = self.shoelace_formula(eye_left) / face_area
        right_area = self.shoelace_formula(eye_right) / face_area
        self.eyes_area = ((self.eyes_area * self.count) + (left_area+right_area)/2) / (self.count + 1)
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

        clf = SVC(class_weight='balanced')
        clf.fit(all_features,all_labels)
        print('Training Error: {}'.format(clf.score(all_features,all_labels)))
        return clf


    def preprocess_frame(self,frame):
        # Crop frame
        frame = frame[:,int(frame.shape[0]/2):]

        # Resize frame for faster detection
        frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
        frame_copy = frame.copy()

        # Adjust contrast and brightness of frame
        # Get histogram for grey scale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        # Get CDF of histogram
        cdf = []
        cdf.append(float(hist[0]))
        [cdf.append(cdf[-1] + float(hist[index])) for index in range(1, len(hist))]

        hist_percent = (cdf[-1]*0.125)
        min_gray = np.argmin(np.abs(np.array(cdf) - hist_percent))
        max_gray = np.argmin(np.abs(np.array(cdf) - cdf[-1] + hist_percent))

        alpha = 255 / (max_gray - min_gray)
        beta = -min_gray * alpha
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        return frame, frame_copy


    def detect(self):
        FEATURES = []
        LABELS = []
        # Read from video and detect head position
        if args.video == '0':
            args.video = 0
        cap = cv2.VideoCapture(args.video)
        assert cap.isOpened(), "Video {} not found".format(args.video)

        # Find size of frame
        _, test_frame = cap.read()
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
            # Train SVM
            clf = self.train_svm()

        # Driver state stores classifications of attentive/inattentive for
        # past 30 frames
        self.driver_state = np.zeros((30,))

        self.count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            frame, frame_copy = self.preprocess_frame(frame) # Adjust frame to make detection easier

            # Feature vector that represents driver state
            # [Eye Area, Trans Vec X, Trans Vec Y, Trans Vec Z]
            current_state = np.array([[0,0,0,0]])

            # Detect face in frame
            face_rects = self.detector(frame, 0)

            head_turned = True
            eyes_area = 0
            if len(face_rects) > 0: # Head detected

                # Get head feature points from first detected head
                shape = face_utils.shape_to_np(self.predictor(frame_copy, face_rects[0]))

                # Find position of head
                rotation_vec, translation_vec, eulerAngles = self.get_head_pose(shape)

                self.count += 1
                for x in range(3): # Euler Angles
                    self.head_rotation[x] += (eulerAngles[x] - self.head_rotation[x]) / self.count
                # print(self.head_rotation)

                # Check if eyes are open
                eyes_area = self.eyes_area_per_face_area(shape)

                # Draw facial features
                head_turned = self.draw_annotation_box(frame_copy,shape,rotation_vec,translation_vec)

                # Create translational vector adjusted for rolling average
                controlled_translational_vec = eulerAngles.flatten() - self.head_rotation

                # Update current driver state
                current_state = np.append(np.array([eyes_area]),
                                        controlled_translational_vec).reshape(1,4)

            # print(self.driver_state.reshape(5,6))

            if args.label:
                FEATURES.append(current_state.flatten())

            # Use SVM to determine if inattentive
            if args.predict or args.simple_predict:
                if args.predict:
                    # 1 if inattentive
                    pred = clf.predict(current_state.reshape(1,-1))
                else:
                    pred = np.array([0])
                    if head_turned or eyes_area < (self.eyes_area / 1.5):
                        pred[0] = 1

                # Update driver state vector
                self.driver_state = np.append(np.delete(self.driver_state,0),pred)
                if np.min(self.driver_state[-20:]) == 1: # Last 10 frames are inattentive
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Frame", frame_copy)
            key = cv2.waitKey(1)
            if args.label or key == 27:
                if key == 27:
                    LABELS.append(0)
                    break
                elif np.max(current_state) == 0:
                    LABELS.append(1)
                elif key == ord('1'):
                    LABELS.append(1)
                else:
                    LABELS.append(0)


        if args.label:
            # Save labeled instances to data file
            index = len(glob.glob('data/features_*'))
            np.save('data/features_'+str(index),FEATURES)
            np.save('data/labels_'+str(index),LABELS)


if __name__ == '__main__':
    # Parse arguments, default to webcam
    parser = argparse.ArgumentParser(description="Pass video file")
    parser.add_argument("--video",dest="video",default="0",
                        help="Path to video file")
    parser.add_argument("--label",dest="label",default=0,
                        help="1 to label video data, labeling is done by \
                        pressing the '1' key while video is played (or in real \
                        time webcam) for inattentive, and the default class is \
                        attentive if no key is pressed for a frame")
    parser.add_argument("--predict",dest="predict",default=0,
                        help="1 to train SVM on given data and predict frames")
    parser.add_argument("--simple_predict",dest="simple_predict",default=0,
                        help="1 to predict inattentiveness based on heuristic")
    args = parser.parse_args()

    # Start detection
    detector = FaceDetector()
    detector.detect()









#

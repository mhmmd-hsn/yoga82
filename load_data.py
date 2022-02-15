import mediapipe as mp
import cv2
import pandas as pd
import requests
import ssl
import glob
import os

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

train_data = "C:\BisotunDev\Yoga-82\yoga_test.txt"  # path to yoga_test.txt

mpPose = mp.solutions.pose
pose = mpPose.Pose()
points = mpPose.PoseLandmark  # Landmarks
data_columns = []
for p in points:
    x = str(p)[13:]
    data_columns.append(x + "_x")
    data_columns.append(x + "_y")
    data_columns.append(x + "_z")
data = pd.DataFrame(columns=data_columns)  # Empty dataset
count = 0
base_6 = []
base_82 = []

# download links
def load_link(url_path):
    with open('pic1', 'wb') as handle:
        response = requests.get(url_path,
                                stream=True)
        if not response.ok:
            print(response)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)


names = []
with open(train_data) as f:
    for line in f:
        x = line.split('/')
        y = line.split(',')
        pose_name = x[0]
        based_on_6 = y[1]
        based_on_82 = y[3]
        exercise_path = 'C:\BisotunDev\Yoga-82\yoga_dataset_links\\' + pose_name + '.txt'  # path to a file in yoga_dataset_links
        if not pose_name in names:
            names.append(pose_name)
            with open(exercise_path) as e:
                
                for line in e:
                    x = line.split()
                    load_link(x[1])
                    if os.path.isfile('pic1'):
                        img = cv2.imread('pic1')
                        try:
                            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            results = pose.process(imgRGB)
                            pose_landmarks = results.pose_landmarks
                            landmarks_to_save = []
                            if results.pose_landmarks:
                                landmarks = results.pose_landmarks.landmark
                                base_6.append(based_on_6)
                                base_82.append(based_on_82)
                                
                                for i, j in zip(points, landmarks):
                                    landmarks_to_save = landmarks_to_save + [j.x, j.y, j.z]
                                data.loc[count] = landmarks_to_save
                                count += 1
                                removing_files = glob.glob('pic1')
                                for i in removing_files:
                                    os.remove(i)
                                print('done')
                        except:
                            print('sth went wrong ...')

        cv2.waitKey(100)
data['based_on_6'] = base_6
data['based_on_82'] = base_82

data.to_csv("dataset3.csv")  # save the data as a csv file

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:13:18 2022

@author: ReneGadeOne
"""
import mediapipe as mp
import cv2
import pickle
import numpy as np
from statistics import mode

# each element's index is its number in 82
yoga82_names = ['Akarna_Dhanurasana', "Bharadvaja's_Twist_pose_or_Bharadvajasana_I_", 'Boat_Pose_or_Paripurna_Navasana_', 'Bound_Angle_Pose_or_Baddha_Konasana_', 'Bow_Pose_or_Dhanurasana_', 'Bridge_Pose_or_Setu_Bandha_Sarvangasana_', 'Camel_Pose_or_Ustrasana_', 'Cat_Cow_Pose_or_Marjaryasana_', 'Chair_Pose_or_Utkatasana_', 'Child_Pose_or_Balasana_', 'Cobra_Pose_or_Bhujangasana_', 'Cockerel_Pose', 'Corpse_Pose_or_Savasana_', 'Cow_Face_Pose_or_Gomukhasana_', 'Crane_(Crow)_Pose_or_Bakasana_', 'Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_', 'Dolphin_Pose_or_Ardha_Pincha_Mayurasana_', 'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_', 'Eagle_Pose_or_Garudasana_', 'Eight-Angle_Pose_or_Astavakrasana_', 'Extended_Puppy_Pose_or_Uttana_Shishosana_', 'Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_', 'Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_', 'Feathered_Peacock_Pose_or_Pincha_Mayurasana_', 'Firefly_Pose_or_Tittibhasana_', 'Fish_Pose_or_Matsyasana_', 'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_', 'Frog_Pose_or_Bhekasana', 'Garland_Pose_or_Malasana_', 'Gate_Pose_or_Parighasana_', 'Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_', 'Half_Moon_Pose_or_Ardha_Chandrasana_', 'Handstand_pose_or_Adho_Mukha_Vrksasana_', 'Happy_Baby_Pose_or_Ananda_Balasana_', 'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_', 'Heron_Pose_or_Krounchasana_', 'Intense_Side_Stretch_Pose_or_Parsvottanasana_', 'Legs-Up-the-Wall_Pose_or_Viparita_Karani_', 'Locust_Pose_or_Salabhasana_', 'Lord_of_the_Dance_Pose_or_Natarajasana_', 'Low_Lunge_pose_or_Anjaneyasana_', 'Noose_Pose_or_Pasasana_', 'Peacock_Pose_or_Mayurasana_', 'Pigeon_Pose_or_Kapotasana_', 'Plank_Pose_or_Kumbhakasana_', 'Plow_Pose_or_Halasana_', 'Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II', 'Rajakapotasana', 'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_', 'Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_', 'Scale_Pose_or_Tolasana_', 'Scorpion_pose_or_vrischikasana', 'Seated_Forward_Bend_pose_or_Paschimottanasana_', 'Shoulder-Pressing_Pose_or_Bhujapidasana_', 'Side-Reclining_Leg_Lift_pose_or_Anantasana_', 'Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_', 'Side_Plank_Pose_or_Vasisthasana_', 'Sitting pose 1 (normal)', 'Split pose', 'Staff_Pose_or_Dandasana_', 'Standing_Forward_Bend_pose_or_Uttanasana_', 'Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_', 'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana', 'Supported_Headstand_pose_or_Salamba_Sirsasana_', 'Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_', 'Supta_Baddha_Konasana_', 'Supta_Virasana_Vajrasana', 'Tortoise_Pose', 'Tree_Pose_or_Vrksasana_', 'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_', 'Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_', 'Upward_Plank_Pose_or_Purvottanasana_', 'Virasana_or_Vajrasana', 'Warrior_III_Pose_or_Virabhadrasana_III_', 'Warrior_II_Pose_or_Virabhadrasana_II_', 'Warrior_I_Pose_or_Virabhadrasana_I_', 'Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_', 'Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_', 'Wild_Thing_pose_or_Camatkarasana_', 'Wind_Relieving_pose_or_Pawanmuktasana', 'Yogic_sleep_pose', 'viparita_virabhadrasana_or_reverse_warrior_pose']

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

#load saved SVC model
first_model = pickle.load(open('first_model.pkl', 'rb'))
loaded_model0 = pickle.load(open('cat_0.pkl', 'rb'))
loaded_model1 = pickle.load(open('cat_1.pkl', 'rb'))
loaded_model2 = pickle.load(open('cat_2.pkl', 'rb'))
loaded_model3 = pickle.load(open('cat_3.pkl', 'rb'))
loaded_model4 = pickle.load(open('cat_4.pkl', 'rb'))
loaded_model5 = pickle.load(open('cat_5.pkl', 'rb'))


#find category based on 6
def get_pose_based_on_6(landmarks):
	prediction_6 = first_model.predict([landmarks])
	print(prediction_6)
	return prediction_6


#find pose based on 82
def current_pose_number(landmarks,based_6):
	if based_6 == 0:
		prediction = loaded_model0.predict([landmarks])
	if based_6 == 1:
		prediction = loaded_model1.predict([landmarks])
	if based_6 == 2:
		prediction = loaded_model2.predict([landmarks])
	if based_6 == 3:
		prediction = loaded_model3.predict([landmarks])
	if based_6 == 4:
		prediction = loaded_model4.predict([landmarks])
	if based_6 == 5:
		prediction = loaded_model5.predict([landmarks])
	return prediction


cap = cv2.VideoCapture('1.m4v')
best_choice = [0,0,0,0,0]
best_6 = [0,0,0,0,0]
#start loop for BlazePose network
mpPose = mp.solutions.pose
while(cap. isOpened()):
	with mpPose.Pose(static_image_mode=False, min_detection_confidence=0.1, model_complexity=1) as pose:
		# Convert the BGR image to RGB and process it with MediaPipe Pose.
		ret, image = cap.read()
		# image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
		results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		pose_landmarks = results.pose_landmarks
		#adjust output image size
		output_increase = 1.5
		image_hight, image_width, _ = image.shape
		enlarge_hight = int(output_increase*image_hight)
		enlarge_width = int(output_increase*image_width)
		large_image = (enlarge_width, enlarge_hight)
		x_end = image.shape[1]
		y_end = image.shape[0]

		if results.pose_landmarks:
			landmarks_to_save = []
			landmarks = results.pose_landmarks.landmark
			annotated_image = image.copy()
			for j in landmarks:
				landmarks_to_save = landmarks_to_save + [j.x, j.y, j.z]

			landmarks_to_save = np.asarray(landmarks_to_save)
			# find main category
			based_on_6 = get_pose_based_on_6(landmarks_to_save)
			based_on_6 = int(based_on_6)
			try:
				best_6.pop(0)
				best_6.append(based_on_6)
				based_on_6 = mode(best_6)
			except:
				pass
			# now find pose
			predicted_pose = current_pose_number(landmarks_to_save, based_on_6)
			predicted_pose = int(predicted_pose)
			# find the best answer based on last 5 frames
			try:
				best_choice.pop(0)
				best_choice.append(predicted_pose)
				predicted_pose = mode(best_choice)
				pose_name = yoga82_names[predicted_pose]
			except:
				pose_name = yoga82_names[predicted_pose]

			cv2.putText(annotated_image, pose_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

			# Draw pose landmarks.
			mp_drawing.draw_landmarks(
				image=annotated_image,
				landmark_list=results.pose_landmarks,
				connections=mpPose.POSE_CONNECTIONS,
				landmark_drawing_spec=drawing_spec,
				connection_drawing_spec=drawing_spec)

			annotated_image = cv2.resize(annotated_image, large_image, interpolation=cv2.INTER_AREA)
			cv2.imshow('Pose', annotated_image)
		else:
			image = cv2.resize(image, large_image, interpolation=cv2.INTER_AREA)
			cv2.imshow('Pose', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# cv2.release()
cv2.destroyAllWindows()
out.release()


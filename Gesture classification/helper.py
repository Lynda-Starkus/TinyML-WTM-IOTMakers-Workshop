import tensorflow as tf
from keras.models import load_model

import numpy as np
import cv2
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


# Path for exported data, numpy arrays
# Step 4
DATA_PATH = os.path.join('ISL_Data')
actions = np.array(['A', 'B', 'C'])

# no of videos
no_sequences = 30

# no of frames in each video
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)

model = load_model('action3.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
SAVED_MODEL = "saved_models"
# tf.saved_model.save(model, SAVED_MODEL)
TFLITE_MODEL = "tflite_models/sign.tflite"
TFLITE_QUANT_MODEL = "tflite_models/sign_quant.tflite"


def TFLite_Predict(test_sample, model_name):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_name)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    pred_shape = (1, 3)

    tflite_interpreter.resize_tensor_input(
        input_details[0]['index'], test_sample.shape)
    tflite_interpreter.resize_tensor_input(
        output_details[0]['index'], pred_shape)
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    test_sample = np.float32(test_sample)
    tflite_interpreter.set_tensor(input_details[0]['index'], test_sample)

    tflite_interpreter.invoke()

    tflite_model_predictions = tflite_interpreter.get_tensor(
        output_details[0]['index'])

    return tflite_model_predictions


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


def realtime(val):
    sequence = []
    sentence = ["Recognized Letter: "]
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw landmarks
            # draw_styled_landmarks(image, results)

            # 2. Prediction logic

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                if val == 0:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]

                if val == 1:
                    res = TFLite_Predict(np.expand_dims(
                        sequence, axis=0), TFLITE_MODEL)[0]

                else:
                    res = TFLite_Predict(np.expand_dims(
                        sequence, axis=0), TFLITE_QUANT_MODEL)[0]

                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

            # Show to screen
            # cv2.imshow('OpenCV Feed', image)

            # Break gracefully
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        # cap.release()
        # cv2.destroyAllWindows()


# realtime()

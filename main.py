import cv2
from util import crop_to_face
import mediapipe as mp
import joblib

#import model
model = joblib.load('mask_detector_LinearSVC.pkl')

#read webcam
webcam = cv2.VideoCapture(0)
#detect faces
mp_face_detection = mp.solutions.face_detection


label = None
penalty_count = 0
bonus_count = 0


penalty_active = False
bonus_active = False

social_credit = 10000

last_bonus_penalty = None




with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.7) as face_detection:
    while True:
        #read webcam frame
        ret,frame = webcam.read()
        #process frame
        img_predict = crop_to_face(frame, face_detection)
        if img_predict is not None and img_predict.size != 0:
            #"preprocessing"
            img_predict = cv2.cvtColor(img_predict, cv2.COLOR_BGR2RGB)
            img_predict = cv2.resize(img_predict, (32, 32))
            img_predict = img_predict.flatten()
            img_predict = img_predict.reshape(1, -1)

            # predict
            prediction = model.predict(img_predict)
            #Handle label and penalty_count
            if prediction[0] == 1:
                label = 'Warning:You will lose credit score'
                penalty_count += 1
                bonus_count = 0
            else:
                label = 'OK'
                penalty_count = 0
                bonus_count += 1
        else:
            label = None
            penalty_count = 0
            bonus_count = 0


        # if exists display label
        if label is not None :
            if label== 'OK':
               if bonus_count >=120:
                   bonus_active = True
                   penalty_active = False
                   social_credit += 100
                   bonus_count = 0
               cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
              if penalty_count >= 90:
                  penalty_active = True
                  bonus_active = False
                  social_credit -= 500
                  penalty_count = 0
              cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if bonus_active:
                cv2.putText(frame, '+100 Social Credit', (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if penalty_active:
                cv2.putText(frame, '-500 Social Credit', (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #credit score = get credit score
            cv2.putText(frame, 'Credit score: {}'.format(str(social_credit)), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        #display frame
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


webcam.release()
cv2.destroyAllWindows()
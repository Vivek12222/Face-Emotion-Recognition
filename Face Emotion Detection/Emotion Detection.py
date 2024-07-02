import cv2 as cv
import numpy as np
from keras.models import load_model
from keras.src.utils.image_utils import img_to_array

emotions_List = {0: "angry", 1: "disgust", 2: "fear", 3: "neutral", 4: "happy", 5: "sad", 6: "surprise"}
model = load_model("D:\\model_weights.h5")  # Change directory according to installed model directory
face_haar_cascade = cv.CascadeClassifier("D:\\OpenCV_Projects\\haar_face.xml")  # Change directory according to installed haar_face.xml directory
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)
        aoi_grays = gray_img[y:y+w, x:x+h]
        aoi_gray = cv.resize(aoi_grays, (48, 48))
        img_pixel = img_to_array(aoi_gray)
        img_pixels = np.expand_dims(img_pixel, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        maxIndex = np.argmax(predictions[0])
        predicted_emotions = emotions_List[maxIndex]
        cv.putText(frame, predicted_emotions, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Facial Emotion Analysis", frame)
    if cv.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

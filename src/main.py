import cv2
from utils import *


log("[ Processing ]\tLoading Model")
model = set_model()
model.load_weights('bin/emotion.h5')
log("[ Success ]\tModel Ready.")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

log("[ Processing ]\tStarting Camera")
try:
    cap = cv2.VideoCapture(0)
    log("[ Success ]\tCamera Ready.")
except Exception:
    log("[ Failed ]\tFailed to start camera.")

img_count= 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    facecasc = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(
        frame, (1280, 720), interpolation=cv2.INTER_CUBIC))

    name = f'logs/img_{datetime.now().strftime("%H%M%S")}_{img_count}.jpg'

    k = cv2.waitKey(1)

    if k % 256 == 32:
        log("[ Processing]\tLogging Frame")
        cv2.imwrite(name,frame)
        log("[ Success ]\tFrame logged.")
        img_count+=1
    if k % 256 == 27:
        log("[ Exiting ]\n")
        break

cap.release()
cv2.destroyAllWindows()

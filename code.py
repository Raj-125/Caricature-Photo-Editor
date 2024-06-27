import cv2
import numpy as np

def radiusWidth(image):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    output = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            output.append(ew)
    return output

def eye_detection(image):

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    output = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            output.append((x + ex + (ew // 2), y + ey + (eh // 2)))
    return output

def bulge(image, center, radius, strength):

    h, w = image.shape[0], image.shape[1]
    warped = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    for i in range(len(center)):
        mapX = np.zeros((h, w), dtype=np.float32)
        mapY = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                dx = x - center[i][0]
                dy = y - center[i][1]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if distance < radius[i]:
                    factor = 1.0 - (distance / radius[i]) ** 2
                    factor = 1 - strength * factor
                    mapX[y, x] = center[i][0] + factor * dx
                    mapY[y, x] = center[i][1] + factor * dy
                else:
                    mapX[y, x] = x
                    mapY[y, x] = y

        warped = cv2.remap(image, mapX, mapY, interpolation=cv2.INTER_LINEAR)
        image = warped
    return warped


image = cv2.imread("test.jpg")
radii = radiusWidth(image)
center = eye_detection(image)
print(center)
image2 = bulge(image, center, radii, 0.5)
cv2.imshow("bulge",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

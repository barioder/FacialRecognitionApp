import cv2
import numpy as np
import face_recognition

# import an image and convert it from BGR2RGB
imgJosh = face_recognition.load_image_file("imagesface/Joshua Cheptegei.jpg")
imgJosh = cv2.cvtColor(imgJosh, cv2.COLOR_BGR2RGB)

# import an image and convert it from BGR2RGB
imgTest = face_recognition.load_image_file("imagesface/Joshua Test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# finding face location and enconding
faceLoc = face_recognition.face_locations(imgJosh)[0]
encodeJosh = face_recognition.face_encodings(imgJosh)[0]
cv2.rectangle(imgJosh, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# comparing faces and finding the distance between them
results = face_recognition.compare_faces([encodeJosh], encodeTest)
faceDis = face_recognition.face_distance([encodeJosh], encodeTest)
print(results, faceDis)

# put text on a window
cv2.putText(imgTest, f"{results} {round(faceDis[0],2)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# display an image in a window
cv2.imshow("Joshua Cheptegei", imgJosh)
cv2.imshow("Joshua Test", imgTest)
cv2.waitKey(0)

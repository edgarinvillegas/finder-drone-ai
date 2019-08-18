# USAGE
# python test_classifier_frame.py
# (See README.md)

from models import CustomClassifier
import cv2
import time

## --- MODEL SELECTION ---
model = CustomClassifier('train_results/mycats_model_3.pt')


# frame = cv2.imread('images/TelloPhoto/juanis-noche-3.png')
## frame = cv2.imread('images/juanis-noche-3-rect.png')         #GRAVE!!
# frame = cv2.imread('images/cat-above-2.jpg')
# frame = cv2.imread('images/juanis-noche-3-rect.png')
# frame = cv2.imread('images/whisky-drone-night-cropped.jpg')
frame = cv2.imread('dataset/valid/juana/juana210.jpg')

#frame = cv2.flip(frame, 0)  # Flip Vertical
#frame = cv2.flip(frame, 1)  # Flip Horizontal
#frame = cv2.GaussianBlur(frame, (11, 11), 0)

start = time.time()
pred = model.predict(frame)

print('Took {} s'.format(time.time()-start))
print(pred)
#cv2.imwrite('images/real-and-toy-cats_output.jpg', frame)
#cv2.imwrite('images/output.jpg', frame)
#cv2.imshow("output", cv2.resize(frame, (1296, 968)))
cv2.imshow("output", frame)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
import cv2

i = 0
for i in range(0, 10):
	cap = cv2.VideoCapture(i)
	test, frame = cap.read()
	print("i : "+str(i)+" /// result: "+str(test))
	del(cap)
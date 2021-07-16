import serial
import time
import sys
import cv2
import os


##---Begin with device set at 70mm X axis and all the way back--##

folder = input("Please Enter Slide Name:  ")
os.mkdir(folder)

COM = 'COM6'# /dev/ttyACM0 (Linux)
BAUD = 115200

ser = serial.Serial(COM, BAUD, timeout = .1)  # for connecting to Arduino
Flag = 0

print('Waiting for device');
time.sleep(3)
print(ser.name)

# Wake up grbl
ser.write(str.encode("\r\n\r\n"))
time.sleep(2)   # Wait for grbl to initialize 
ser.flushInput()  # Flush startup text in serial input

# Wake up camera
cam = cv2.VideoCapture(1)
return_value, image = cam.read()
cv2.imwrite(folder+ '/wakeup.jpg', image)

print('Set Absolute Distance Mode')
ser.write(str.encode('G91\n'))
grbl_out = ser.readline() # Wait for grbl response with carriage return
print (grbl_out.strip())

print('Go to Zero')
y = 1
p = 0
print ('(' + '1' + ',' + str(y) + ')')
ser.write(str.encode('G0 X0 Y0 Z0 \n'))
grbl_out = ser.readline() # Wait for grbl response with carriage return
print (grbl_out.strip())
name = str(p)
name = name.rjust(6,'0')
cv2.imwrite(folder + '/' + name + '.jpg', image)
p = p+1


for k in range(1,6): # Set Y moves here = (total moves/2) +1 = (10/2)+1 = 6

	for i in range(2,17):  ## Set X moves here = total moves + 1 = 16+1 = 17
		print('(' + str(i) + ',' + str(y) + ')')
		ser.write(str.encode('G0 X-3 Y0 Z0 \n'))  #Set X increment here in mm (-3 default)
		grbl_out = ser.readline()
		print (grbl_out.strip())
		time.sleep(1.5)
		name = str(p)
		name = name.rjust(6,'0')
		cv2.imwrite(folder + '/' + name + '.jpg', image)
		p = p+1

	y = y+1    
	print ('(' + '9' + ',' + str(y) + ')')
	ser.write(str.encode('G0 X0 Y2.2 Z0 \n'))  #Set Y increment here in mm (2.2 default)
	grbl_out = ser.readline()
	print(grbl_out.strip())
	time.sleep(1.5)
	name = str(p)
	name = name.rjust(6,'0')
	cv2.imwrite(folder + '/' + name + '.jpg', image)
	p = p+1

	for j in range(2,17):
		print('(' + str(17-j) + ',' + str(y) + ')')
		ser.write(str.encode('G0 X3 Y0 Z0 \n'))  #Set X increment again
		grbl_out = ser.readline()
		print(grbl_out.strip())
		time.sleep(1.5)
		name = str(p)
		name = name.rjust(6,'0')
		cv2.imwrite(folder + '/' + name + '.jpg', image)
		p = p+1

	y = y+1    
	print ('(' + '1' + ',' + str(y) + ')')
	ser.write(str.encode('G0 X0 Y2.2 Z0 \n'))  #Set Y increment again
	grbl_out = ser.readline()
	print (grbl_out.strip())
	time.sleep(1.5)
	name = str(p)
	name = name.rjust(6,'0')
	cv2.imwrite(folder + '/' + name + '.jpg', image)
	p = p+1

ser.write(str.encode('G0 X0 Y-22 Z0 \n'))  #Set return increments here.  = y increments*total moves = 2.2*10

del cam
# Wait here until grbl is finished to close serial port and file.
input("  Press <Enter> to exit and disable grbl.") 

# Close file and serial port
ser.close() 

def takePhotoWithDelay(p, folder, image):
	grbl_out = ser.readline()
	print (grbl_out.strip())
	time.sleep(1.5)
	name = str(p)
	name = name.rjust(6,'0')
	cv2.imwrite(folder + '/' + name + '.jpg', image)
	p = p+1

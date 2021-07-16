import serial
import time
import sys
import cv2
import os

'''
Author: Junbong Jang
Date: 6/10/2019

Refactored Brian WestGate's MasterPhotoCapture code into more succint, readable and maintainable code.
This takes a photo after each movement of the sample slide a little bit left/right/up.
With 2x nosepiece light microscope, It takes 170 images in total.
'''


class PhotoCapture:

    # Begin with device set at 70mm X axis and all the way back--##
    def __init__(self, folder_name):
        self.image_counter = 0
        self.folder_name = folder_name
        COM = 'COM6'  # /dev/ttyACM0 (Linux)
        BAUD = 115200
        self.ser = serial.Serial(COM, BAUD, timeout=.1)  # for connecting to Arduino

        print('Waiting for device')
        time.sleep(3)
        print(self.ser.name)

        # Wake up grbl
        self.ser.write(str.encode("\r\n\r\n"))
        time.sleep(2)  # Wait for grbl to initialize
        self.ser.flushInput()  # Flush startup text in serial input

        # Wake up camera
        self.cam = cv2.VideoCapture(1)
        return_value, self.image = self.cam.read()
        cv2.imwrite(self.folder_name + '/wakeup.jpg', self.image)

        print('Set Absolute Distance Mode')
        self.ser.write(str.encode('G91\n'))
        print(self.ser.readline().strip())  # Wait for grbl response with carriage return

        print('Go to Zero')
        self.ser.write(str.encode('G0 X0 Y0 Z0 \n'))

    def capturePhotos(self):
        y = 1
        for k in range(1, 11):  # Set Y moves here = (total moves/2) +1 = (10/2)+1 = 6

            self.capturePhotoWithDelay()

            for i in range(1, 16):  # Set X moves here = total moves + 1 = 16+1 = 17
                if i == 1:
                    if k % 2 == 1:
                        serial_move_command = 'G0 X-3.6 Y0 Z0 \n'
                    else:
                        serial_move_command = 'G0 X4.0 Y0 Z0 \n'
                else:
                    if k % 2 == 1:
                        serial_move_command = 'G0 X-3 Y0 Z0 \n'
                    else:
                        serial_move_command = 'G0 X3 Y0 Z0 \n'

                print('(' + str(i) + ',' + str(y) + ')')
                print(self.ser.readline().strip())
                self.ser.write(str.encode(serial_move_command))  # Set X increment here in mm (-3 default)
                self.capturePhotoWithDelay()

            y = y + 1
            self.ser.write(str.encode('G0 X0 Y2.2 Z0 \n'))  # Set Y increment here in mm (2.2 default)

    def capturePhotosPrecise(self):
        y = 1
        for k in range(1, 101):  # Set Y moves here = (total moves/2) +1 = (10/2)+1 = 6

            self.capturePhotoWithDelay()

            for i in range(1, 161):  # Set X moves here = total moves + 1 = 16+1 = 17
                if i == 1:
                    if k % 2 == 1:
                        serial_move_command = 'G0 X-.36 Y0 Z0 \n'
                    else:
                        serial_move_command = 'G0 X.40 Y0 Z0 \n'
                else:
                    if k % 2 == 1:
                        serial_move_command = 'G0 X-.3 Y0 Z0 \n'
                    else:
                        serial_move_command = 'G0 X.3 Y0 Z0 \n'

                print('(' + str(i) + ',' + str(y) + ')')
                print(self.ser.readline().strip())
                self.ser.write(str.encode(serial_move_command))  # Set X increment here in mm (-3 default)
                self.capturePhotoWithDelay()

            y = y + 1
            self.ser.write(str.encode('G0 X0 Y.22 Z0 \n'))  # Set Y increment here in mm (2.2 default)

    def terminateDevice(self):
        del self.cam

        self.ser.write(str.encode('G0 X0 Y-22.7 Z0 \n'))  # Set return increments here. y increments*total moves = 2.2*10
        # Wait here until grbl is finished to close serial port and file.
        input("  Press <Enter> to exit and disable grbl.")
        # Close file and serial port
        self.ser.close()

    def capturePhotoWithDelay(self):
        time.sleep(1.7)
        # name = str(self.image_counter).rjust(6, '0')
        # cv2.imwrite(self.folder_name + '/' + name + '.jpg', self.image)
        # self.image_counter += 1

    def capturePhotoAutomatically(self):
        self.capturePhotos()
        self.terminateDevice()


if __name__ == '__main__':
    folder_name = ''
    # folder_name = '../assets/' + input("Please Enter Slide Name:  ")
    # os.mkdir(folder_name)
    photo_capture_obj = PhotoCapture(folder_name)
    photo_capture_obj.capturePhotoAutomatically()

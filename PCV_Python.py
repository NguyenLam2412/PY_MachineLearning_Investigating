import cv2
import numpy
import os

# img = numpy.array([[250,252,253],[104,105,206],[107,118,129]])
# print(img)
# print(img.shape)
######################################################
# img = cv2.imread('empire.jpg')
# cv2.imwrite('MyPic1.png', img)
######################################################
# Make an array of 5 random bytes
# randomByteArray = bytearray(os.urandom(45))
# flatNumpyArray = numpy.array(randomByteArray)
# print(flatNumpyArray)
# numpy.random.randint(0,256,45)
# Convert the array to make a 5x5 grayscale image
# grayImage = flatNumpyArray.reshape(5,9)
# print(grayImage)
# cv2.imwrite("RandomGray.jpg", grayImage)

# # Convert the array to make a 5x3 color image
# brgImage = flatNumpyArray.reshape(5,3,3)
# print(brgImage)
# cv2.imwrite("RandomColor.jpg", brgImage)

# # Convert to gray again
# grayImage1 = cv2.cvtColor(brgImage,cv2.COLOR_BGR2GRAY)
# print(grayImage1)
# cv2.imwrite("RandomGray1.jpg", grayImage1)
######################################################
# img = cv2.imread('RandomColor.jpg')
# print(img)
# # print(img.item(1,1,0))  # Prints the current value of B  
# # img.itemset((1,1,0),0)
# img[:,:,1] = 0
# print(img)
# cv2.imwrite("ChangedColor.jpg", img)
#####################################################
# clicked = False
# def onMouse(event,x ,y, flags, param):
#     global clicked
#     if event == cv2.EVENT_LBUTTONUP:
#         clicked = True
# cameraCapture = cv2.VideoCapture(0)
# cv2.namedWindow('MyWin')
# cv2.setMouseCallback('MyWin',onMouse)
# print('Showing camera feed. Click window or press any key to stop')
# success, frame = cameraCapture.read()
# while success and cv2.waitKey(1) == -1 and not clicked:
#     cv2.imshow('Mywin', frame)
#     success, frame = cameraCapture.read()
# cv2.destroyWindow('MyWin')
# cameraCapture.release()
# cv2.destroyAllWindows()
# cv2.waitKey(0)

img = cv2.VideoCapture(0)
frame = img.grab()
print(frame)
# while cv2.waitKey(0) != -1:
#     cv2.destroyAllWindows() 

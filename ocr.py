# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
# from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def image_smooth(img):
	ret1, th1 = cv2.threshold(img,127, 255, cv2.THRESH_BINARY)
	ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	blur = cv2.medianBlur(th2,3)
	ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return th3

def image_smooth2(img):
	ret1, th1 = cv2.threshold(img,180, 255, cv2.THRESH_BINARY)
	ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	blur = cv2.GaussianBlur(th2,(1, 1), 0)
	ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return th3

def mser(image):
	#Detecting Text Regions
	#Create MSER object
	mser = cv2.MSER_create()

	vis = image.copy()

	regions, _ = mser.detectRegions(image)
	#Use to extend the text regions to extend over sharp turns
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

	cv2.polylines(vis, hulls, 1, (0, 255, 0))

	cv2.imshow('img', vis)

	mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

	for contour in hulls:
		cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

	#this is used to find only text regions, remaining are ignored
	text_only = cv2.bitwise_and(image, image, mask=mask)

	cv2.imshow("text only", text_only)
	cv2.waitKey(0)
	return text_only

# def contours(img): 
	
# 	gray1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	ret6, mask = cv2.threshold(gray1, 180, 255, cv2.THRESH_BINARY)
# 	image_final = cv2.bitwise_and(gray1, gray1, mask=mask)
# 	ret6, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  
# 	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
# 															 3))  
# 	# to manipulate the orientation of dilution , 
# 	dil = cv2.dilate(new_img, kernel, iterations=5)
# 	#Adding Contours
# 	_, contours, hierarchy = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 	for contour in contours:
# 		[x, y, w, h] = cv2.boundingRect(contour)
# 		# draw rectangle around contour on original image
# 		if w < 35 and h < 35:
# 			pass
# 		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
# 		cropped = image_final[y :y +  h , x : x + w]
# 		cv2.imshow('contours',img)
# 		cv2.waitKey(0)
# 		cv2.imshow('cropped',img)
# 		cv2.waitKey(0)
# 		return img


def file1(txt):
	#Writing data in the file
	f=open("C:\\Users\\HP\\Documents\\Image Project\\image.txt","w+")
	f.write(txt)
	f.close()
	return 0

def x_values(h,w):
	if h<=200 and w<=300:
		x=2.5
	elif h<=500 and w<600:
		x=2.0
	elif h<=600 and w<=800:
		x=1.4
	else:
		x=1
	return x


def y_values(h,w):
	if h<=200 and w<=300:
		y=2.5
	elif h<=500 and w<600:
		y=2.0
	elif h<=600 and w<=800:
		y=1.4
	else:
		y=1
	return y


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="specify path to input image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

#Resize Parameters
image1 = cv2.imread(args["image"])
im = Image.open(args["image"])
width, height = im.size
print(height)
print(width)

x1=x_values(height,width)
y1=y_values(height,width)

 
image = cv2.resize(image1, None, fx=x1, fy=y1, interpolation=cv2.INTER_CUBIC)

gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if args["preprocess"] == "text":
	#Contours
	gray1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(gray1, 180, 255, cv2.THRESH_BINARY)
	image_final = cv2.bitwise_and(gray1, gray1, mask=mask)
	ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
															 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
	dilated = cv2.dilate(new_img, kernel, iterations=9)  
	# dilate , more the iteration more the dilation
	_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
	# findContours returns 3 variables for getting contours

	for contour in contours:
		[x, y, w, h] = cv2.boundingRect(contour)

		# Don't plot small false positives that aren't text
		if w < 35 and h < 35:
			continue
		# draw rectangle around contour on original image
		cv2.rectangle(image1, (x, y), (x + w, y + h), (255, 0, 255), 2)
	cv2.imshow('contour',image1)
	cv2.waitKey(0)

	#mser
	img2 = mser(gray)
	#inverting the grayscale image
	new=image_smooth2(gray)
	ret4, th4 = cv2.threshold(new,155, 255, cv2.THRESH_BINARY)
	#Canny Edge Detection
	edges = cv2.Canny(th4,100,200)
	cv2.imshow('edge',edges)
	cv2.waitKey(0)

	#Pre-Processing
	filtered = cv2.adaptiveThreshold(edges.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
	kernel = np.ones((1, 1), np.uint8)
	opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	or_image = cv2.bitwise_or(th4, closing)
	ret5, th5 = cv2.threshold(or_image,140, 255, cv2.THRESH_BINARY)
	cv2.imshow('pre',or_image)
	cv2.waitKey(0)
	cv2.imshow('next',th5)                                                                                                                                                                               

	#Print the filenames
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename,th5)

	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print(text)
	file1(text)
	cv2.waitKey(0)


elif args["preprocess"]=="thresh":
	#Contours
	gray1= cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(gray1, 180, 255, cv2.THRESH_BINARY)
	image_final = cv2.bitwise_and(gray1, gray1, mask=mask)
	ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
															 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
	dilated = cv2.dilate(new_img, kernel, iterations=9)  
	# dilate , more the iteration more the dilation
	_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
	# findContours returns 3 variables for getting contours

	for contour in contours:
		[x, y, w, h] = cv2.boundingRect(contour)

		# Don't plot small false positives that aren't text
		if w < 35 and h < 35:
			continue
		# draw rectangle around contour on original image
		cv2.rectangle(image1, (x, y), (x + w, y + h), (255, 0, 255), 2)
	cv2.imshow('contour',image1)
	cv2.waitKey(0)
	
	#mser
	img2= mser(gray)
	#inverting the grayscale image
	#Canny Edge Detection
	edges = cv2.Canny(gray,100,200)
	cv2.imshow('edge',edges)
	cv2.waitKey(0)

	#Pre-Processing
	filtered = cv2.adaptiveThreshold(edges.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
	kernel = np.ones((1, 1), np.uint8)
	opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	new1=image_smooth(gray)
	or_image = cv2.bitwise_or(new1, closing)
	cv2.imshow('pre',or_image)

	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, or_image)

	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print(text)
	file1(text)
	cv2.waitKey(0)

elif args["preprocess"]=="option":
	
	ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	img2= mser(gray)
	#hsv= cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
	#Canny Edge Detection
	cv2.imshow('mser',img2)
	cv2.waitKey(0)
	#Preprocessing for colored backgrounds
	'''
	hsv= cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
	cv2.imshow('hsv',hsv)
	cv2.waitKey(0)
	'''
	filtered = cv2.adaptiveThreshold(img2.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
	kernel = np.ones((1, 1), np.uint8)
	opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	new1=image_smooth(gray)
	or_image = cv2.bitwise_or(new1, closing)
	cv2.imshow('pre',or_image)

	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, or_image)

	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print(text)
	file1(text)
	cv2.waitKey(0)


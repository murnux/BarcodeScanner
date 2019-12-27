# binarizes the image and does some cropping, uses erosion for edge detection. Does not analyze the bars

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# General Approach and Methods:
## Binarize the image, increase / decrease contrast, increase / decrease brightness
## shape detection
## use the above shape detection to 'estimate' width value


# binarizeImg takes in an image and converts it to binary using a passed in threshold
def binarizeImg(thresh, baseImg, basePix): 
  newImg = Image.new('L', (baseImg.width, baseImg.height)) # make a copy as to not change our original image
  newPix = newImg.load()
  for i in range(baseImg.width):
    for j in range(baseImg.height):
      newPix[i, j] = basePix[i, j] # copy our image

  for i in range(newImg.width):
    for j in range(newImg.height):
      if newPix[i, j] < thresh: # check against the threshold value
        newPix[i, j] = 0
      else:
        newPix[i, j] = 1
  return newImg

# function that increases the contrast of an image given an intensity
def adjustContrast(baseImg, intensity):
  contrast = ImageEnhance.Contrast(baseImg)
  contrast = contrast.enhance(intensity) # set FACTOR > 1 to enhance contrast, < 1 to decrease
  return contrast

# using a modifier, returns an image with a changed brightness setting
def adjustBrightness(baseImg, modifier):
  pixels = baseImg.load()
  for i in range(baseImg.width):
    for j in range(baseImg.height):
      newValue = pixels[i, j] + modifier
      if newValue > 255:
        pixels[i, j] = 255
      elif newValue < 0:
        pixels[i, j] = 0
      else:
        pixels[i, j] = newValue
  
  return baseImg

# calculates the min and max pxiels in an image, uses the difference to
# determine whether the quality of the image is good or poor
def imgIsGood(img, thresh):
  npArray = np.asarray(img) # convert to numoy array for easier calculation
  min = np.amin(npArray) # get the minimum pixel value
  max = np.amax(npArray) # get the maximum pixel value

  if min == 0 and max == 255: # if both min and max is 0 and 255, probably a digital barcode image
    return True
  else: # otherwise likely a photo of a bracode
    if (max - min) > thresh:
      return False
    else:
      return True

# SMOOTH AND SHARPEN KERNELS GRABBED FROM: https://www.taylorpetrick.com/blog/post/convolution-part3
def smoothImage(img):
  kernel = np.ones((4,4),np.float32)/25 # smooth kernel
  adjusted = cv2.filter2D(np.asarray(img),-1,kernel)
  return Image.fromarray(adjusted)

def sharpenImage(img):
  kernel = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")
  adjusted = cv2.filter2D(np.asarray(img),-1,kernel)
  return Image.fromarray(adjusted)

def rotateImage(image, angle):
  imageCenter = tuple(np.array(image.shape[1::-1]) / 2)
  rotationMatrix = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)
  result = cv2.warpAffine(image, rotationMatrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# reverses the colors of a binary image
def inverseImage(arr):
  #return np.invert(arr)
  for i in range(len(arr)):
    for j in range(len(arr[i])):
      if arr[i, j] == 0:
        arr[i, j] = 1
      else:
        arr[i, j] = 0
  return arr


def detectBarCode(origImg, binImg):
  # finds all of the contours of the image of all detected shapes
  contours, _ = cv2.findContours(binImg,1,2)
  img = binImg # copy our binImg into a new variable to not change the original binarized version
  validBar = [] # valid bar consists of the coordinates of all rectangles detected if it is assumed to be a bar in the barcode
  maxAngle = 0
  for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt) # gets the x, y coords and size of the contour
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx)==4: # if approx is 4, this contour is a rectangle / square
      
      # with x and y > 0, skip the edge of the image being considered a contour.
      # h and w calculations help assure that we only grab contours that are rectangles
      # in the barcode and not from anywhere else in the image
      if x > 0 and y > 0 and h > 30 and w < 70:
        # If all above are true, then this is probably a barcode rectangle. Add it as a valid contour.
        rotatedRect = cv2.minAreaRect(cnt)
        validBar.append((x, y, x + w, y + h))
        angle = rotatedRect[2] * -1 # stores the angle of the contour
        if angle > maxAngle and angle != -90:
          maxAngle = angle 

  validNp = np.asarray(validBar) # make validBars into a numpy array
  left = np.min(validNp[:,0])
  top = np.min(validNp[:,1])
  right = np.max(validNp[:,2])
  bottom = np.max(validNp[:,3])

  drawBoundingBox(np.asarray(origImg), contours, (left, top, right, bottom))
  rotated = rotateImage(binImg, (maxAngle * -1) // 2) # use our maxAngle // 2 as the amount to rotate

  # convert to pillow image for simple cropping, then return back the numpy array value of the cropped image
  image = Image.fromarray(rotated)
  image = image.crop((left, top, right, bottom))
  return np.asarray(image) 

def drawBoundingBox(image, contours, points):
  # following gets coordinates needed for cropping
  rectImage = np.asarray(image.copy())
  contImage = np.asarray(image.copy())

  cv2.drawContours(contImage, contours, -1, (255, 0, 0), 2)
  print("DETECTED CONTOURS")
  saveArrayAsImage("contours.jpg", contImage)
  
  cv2.rectangle(rectImage, (points[0], points[1]), (points[2], points[3]), (255, 0, 0), 3)
  print("DETECTED BARCODE REGION")
  saveArrayAsImage("region.jpg", rectImage * 255)

def saveArrayAsImage(filename, array):
  arrayAsImg = Image.fromarray(array)
  arrayAsImg.save('./results/' + filename)

# from: https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def get_four_most_common_list_values(my_list):
  most_common = Counter(my_list)
  most_common = most_common.most_common(4)
  to_return = []
  for i in most_common:
    to_return.append(i[0])
  return to_return

def convert_common_to_widths(my_list, most_common):
  new_list = []
  for item in range(len(my_list)):
    for common in range(len(most_common)):
      if my_list[item] == most_common[common]:
        new_list.append(common + 1) # bar width values can be 1, 2, 3, 4. 
  return new_list

# follows the standard of 4 bar width values equaling one bar code digit. Sets this idea up.
def convert_widths_to_tuple(widths):
  tuple_list = []
  single_item = []
  counter = 0
  print(len(widths))
  for item in range(len(widths)):
    if item == counter + 4:
      tuple_list.append(single_item)
      single_item = []
      counter = item + 1
    else:
      single_item.append(widths[item])

  return tuple_list

# Uses the barcode values seen here: https://www.wikihow.com/Read-12-Digit-UPC-Barcodes
# converts the width values taken in the image and assign a numeric value
# uses the margin of error to assign a value even if a width value is slightly off
def assign_values_to_widths(widths, margin):
  actual_values = []
  width_values =	{
    3211: 0,
    2221: 1,
    2122: 2,
    1411: 3,
    1132: 4,
    1231: 5,
    1114: 6,
    1312: 7,
    1213: 8,
    3112: 9,
  }

  for width in widths:
    number = ''.join(map(str, width)) # convert the array of digits into a single number as a string
    number = int(number) # use a cast to convert our number string into an integer
    for value in width_values.keys():
      if number == value:
        actual_values.append(width_values[value])
      else:
        for m in range(number, number + margin): # account for margin of error in positive direction
          if m == value:
            actual_values.append(width_values[value])
            continue
        for m in range(number - margin, number): # account for margin of error in negative direction
          if m == value:
            actual_values.append(width_values[value])
            continue

  return actual_values
        

# takes in a numpy array representation of an image and attempts to analyze the barcode
def analyzeBarCode(image):
  middle = len(image) // 2
  values = [] # used to store the perceived numerical value of the barcodes
  count = 0
  currentBar = 0
  for pixel in range(len(image[middle])):
    pixelVal = image[middle][pixel]
    if pixel == 0: # if this is the first iteration
      currentBar = pixelVal
    else:
      if pixelVal == currentBar:
        count += 1
      else:
        currentBar = pixelVal
        values.append(count)
        count = 0

  values = remove_values_from_list(values, 0)  
  common = get_four_most_common_list_values(values)
  print("common:", common)
  only_most_common = []
  #for ctr in range(len(values)):
   # if values[ctr] in common:
    #  only_most_common.append(values[ctr])

  widths = convert_common_to_widths(values, common)
  tuples = convert_widths_to_tuple(widths)
  print("tuples:", tuples)

  new_values = assign_values_to_widths(tuples, 20)
  print("new values:", new_values)

  return values
    

#### BEGIN MAIN ####

# load the barcode
imOrig = Image.open('barcode.png')
im = imOrig.convert('L') # convert to grayscale
im.save("./results/original.jpg")

erosionFactor = 0 # will change based on whether it is a photo or digital scan
# if image is not a digital scan of a barcode, meaning it is probably a photo, perform some cleanup.
if not imgIsGood(im, 100):
  print("This image seems to be a photo of a barcode, attempting to clean up and binarize.")
  im = adjustBrightness(im, -40) # increasing brightness seems to make fabrizzio's picture worse
  #im = adjustContrast(im, -100)
  #im = smoothImage(im)
  im = sharpenImage(im)
  saveArrayAsImage("enhanced.jpg", np.asarray(im))
  erosionFactor = 6
else:
  print("Image seems to be a digital scan. Going to binarize without cleaning up")
  erosionFactor = 3

# convert the image to binary, im will now hold the binarized pixels
im = binarizeImg(127, im, im.load())

# store im as a numpy array, makes it easier to perform operations
npBin = np.asarray(im)
# attempt to draw and crop the image to where the barcode is
npBin = detectBarCode(im, npBin)

# build a convolution kernel, and use it to generate the difference between the image and its eroded version
kernel = np.ones((erosionFactor, erosionFactor),np.uint8)
erosion = cv2.erode(npBin, kernel)

difference = npBin - erosion # use erosion for edge detection

## Save our final image to be used for analysis

difference = inverseImage(difference)

saveArrayAsImage("final.jpg", difference * 255)

data = analyzeBarCode(difference)
print("data:", data)
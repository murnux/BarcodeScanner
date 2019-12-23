# binarizes the image and does some cropping, uses erosion for edge detection. Does not analyze the bars

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import cv2

# General Approach and Methods:
## Binarize the image, increase / decrease contrast, increase / decrease brightness
## shape detection
## use the above shape detection to 'estimate' width value


# binarizeImg takes in an image and converts it to binary using a passed in threshold
def binarizeImg(thresh, baseImg, basePix): 
  newImg = Image.new('L', (baseImg.width, baseImg.height)) # make a copy as to not change our original image
  newPix = newImg.load()
  npArray = np.asarray(newImg)
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
  plt.imshow(contImage)
  plt.show()
  
  cv2.rectangle(rectImage, (points[0], points[1]), (points[2], points[3]), (255, 0, 0), 3)
  print("DETECTED BARCODE REGION")
  plt.imshow(rectImage)
  plt.show()

#### BEGIN MAIN ####

# load the barcode
imOrig = Image.open('barcode.jpg')
im = imOrig.convert('L') # convert to grayscale

print("Original image in grayscale for comparison:")
plt.imshow(im, cmap="gray")
plt.show()

erosionFactor = 0 # will change based on whether it is a photo or digital scan
# if image is not a digital scan of a barcode, meaning it is probably a photo, perform some cleanup.
if not imgIsGood(im, 50):
  print("This image seems to be a photo of a barcode, attempting to clean up and binarize.")
  im = adjustBrightness(im, -30) # increasing brightness seems to make fabrizzio's picture worse
  im = adjustContrast(im, -100)
  #im = smoothImage(im)
  im = sharpenImage(im)
  erosionFactor = 6
else:
  print("Image seems to be a digital scan. Going to binarize without cleaning up")
  erosionFactor = 3

# convert the image to binary, im will now hold the binarized pixels
im = binarizeImg(127, im, im.load())

# store im as a numpy array, makes it easier to perform operations
npBin = np.asarray(im)
# attempt to draw and crop the image to where the barcode is
npBin = detectBarCode(imOrig, npBin)

# build a convolution kernel, and use it to generate the difference between the image and its eroded version
kernel = np.ones((erosionFactor, erosionFactor),np.uint8)
erosion = cv2.erode(npBin, kernel)
print("BINARY")
plt.imshow(npBin, cmap='gray', vmin=0,vmax=1)
plt.show()
print("EROSION")
plt.imshow(erosion, cmap='gray', vmin=0,vmax=1)
plt.show()

difference = npBin - erosion # use erosion for edge detection

## Save our final image to be used for analysis
imNewFromArray = Image.fromarray(difference)
imNewFromArray.save('test.png')

print("FINAL EDGE DETECTION IMAGE USED FOR ANALYSIS (Binary - Erosion)")
plt.imshow(difference, cmap='gray', vmin=0,vmax=1)
plt.show()

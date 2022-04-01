# Exercise 1
As part of Computer Vision and Image Processing Course we need to write some functions that allow us to perform some operations on images.

##### Technical issues:
Python 3.10\
Numpy 1.22.3\
Cv2 4.5.5\

## Files
### 1) ex_utils.py
The main file that contains all logic (expect gamma correction).
#### Functions:
* **imReadAndConvert(filename: str, representation: int) -> np.ndarray**\
  Get a path to image and return it as a ndarray RGB or GRAYSCALE
  <br />
  

* **imDisplay(filename: str, representation: int)**\
    Get a path to image and show it as required, RGB or GRAYSCALE respectively
  <br />
  
  
* **transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray**\
    Transform RGB image to YIQ
  <br />
  

* **transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray**\
    Transform YIQ image to RGB
  <br />
  

* **collect_data(imgOrig: np.ndarray) -> np.ndarray**\
    Get an image as RGB and extract the Y channel from it (after transformation).\
  If the image is in GRAY SCALE mode just return it as is
  <br />
  

* **back_to_image(imgOrig: np.ndarray, data: np.ndarray) -> np.ndarray**\
   Get an image before processing, and the Y channel after processing and connect them to RGB image.\
  If the original image was in GRAY SCALE mode so return the processed image as is.
  <br />
  

* **hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray)**\
    Get an image and repair it by equalize histogram principles.
  <br />
  

* **quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float])**\
    Get an image and quantized it to `nQuant` colors within `nIter` iterations
      <br />
  

  
### 2) gamma.py
This file had the code about gamma correction.
#### Function:
* **gammaDisplay(img_path: str, rep: int)**\
Get a path to an image and show the image in RGB or GRAY SCALE mode.\
  In addition show GUI such u can handle the gamma value between 0 to 2 with steps of 0.01
  <br />
  

### 3) ex_utils.py
   This the main that use the other files and run all functions above.
### 4) testImg1.jpg
![My image](testImg1.jpg)
  <br />
  
### 5) testImg2.jpg
![My image](testImg2.jpg)
  <br />
  

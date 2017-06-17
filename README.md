

# **Vehicle Detection**

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./WRITEUP/images/hog_img.png
[image3]: ./WRITEUP/images/hog-sub.jpg
[image4]: ./WRITEUP/images/sliding_window.png
[image5]: ./WRITEUP/images/bboxes_and_heat.png
[image51]: ./WRITEUP/images/test_image_4+heat.png
[image52]: ./WRITEUP/images/test_image_5+heat.png
[image53]: ./WRITEUP/images/test_image_6+heat.png
[image6]: ./WRITEUP/images/labels.png
[image7]: ./WRITEUP/images/image_with_debug.png
[video1]: ./WRITEUP/video/P5_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and tried to keep the HOG feature vector to a minimum as to keep the processing time low. Having a fast processing speed will be required when the program is running in real time on a car. This combination of parameters seemed to work and generated a lower number of cells / smaller feature vector length.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Using the selected HOG and color features I trained the SVM to have a test accuracy of 99%. The feature vector length was 1284. The features were appended into a single vector and then normalized so that the histogram and HOG features were scaled to a zero mean and unit variance. 

The code is shown below.

```python
car_features, noncar_features = [], []

car_features = extract_features(car_imgs, color_space='YUV', orient=orient, pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel='ALL', spatial_feat=False, hist_feat=True,hog_feat=True)

noncar_features = extract_features(notcar_imgs, color_space='YUV', orient=orient, pix_per_cell=pix_per_cell, cell_per_block=2, hog_channel='ALL', spatial_feat=False, hist_feat=True,hog_feat=True)

car_features = np.asarray(car_features)
noncar_features = np.asarray(noncar_features)

# Create an array stack of scaled feature vectors
unscaled_x = np.vstack((car_features, noncar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(unscaled_x)
# Apply the scaler to X
scaled_X = X_scaler.transform(unscaled_x)
y = np.hstack((np.ones(len(car_features)),np.zeros(len(noncar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))

from sklearn.svm import LinearSVC
# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
print('My SVC predicts: ', svc.predict(X_test[0].reshape(1, -1)))
print('For labels: ', y_test[0])
```

`(14208, 1284) (14208,)
(3552, 1284) (3552,)
Using: 11 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1284
Test Accuracy of SVC =  0.994087837838
My SVC predicts:  [ 0.]
For labels:  0.0`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To save computation time I limited the search windows to below the horizon. (There should not be any cars in the sky...) I then used 3 search windows of nxn size (square) to search for and identify the cars. The search window sizes were: 

* 64x64 pixels
* 96x96 pixels
* 128x128 pixels

These were chosen because they were all clean multiples of the original training window size of 64x64 pixels. 

An example of the search area is shown below: 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image41]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./WRITEUP/video/P5_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  

To threshold the heatmap, I que'd up 24 frames and summed them all up. If an area had 13 or more positive detections in the que, it would pass through the threshold. Assuming the dash-cam was recording at ~24 frames per second, this would equate to a half-second window filter which was good enough to block out a lot of the false positives. 

I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image51]
![alt text][image52]
![alt text][image53]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all window sizes:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is most likely to fail in less clear (night/ evening/ rain) lighting conditions. Adding more images that covered these other circumstances could help to improve in these conditions. 

Another improvement to improve scan speed would be to further decrease the search window. If you can successfully identify a lane or a highway divider you might be able to decrase the scan window; thus increasing the speed and efficiency of the algorithm. You also wouldn't detect cars over the divider, thereby further reducing false positives. 



```python

```

# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

## Results

A total of 140 tests were executed

GPU

Average timing:
1. dog.jpeg = 0.061996 ms
2. dog2.jpeg = 0.0306167 ms
3. dog3.jpeg = 0.0292741 ms 
4. scenery.jpg = 0.0232988 ms
5. woman.jpg = 0.0454346 ms
6. woman2.jpg = 0.035535 ms 
7. woman3.jpg = 0.0316561 ms

Total average timing: 0.0368301857 ms

CPU

Average timing:
1. dog.jpeg = 256.4787612 ms
2. dog2.jpeg = 285.4946318 ms
3. dog3.jpeg = 280.9534698 ms
4. scenery.jpg = 11.725012 ms
5. woman.jpg = 237.5207017 ms
6. woman2.jpg = 232.6109392 ms 
7. woman3.jpg = 231.8336227 ms

Total average timing: 213.3563962 ms

Speed up: 5960.2396729701

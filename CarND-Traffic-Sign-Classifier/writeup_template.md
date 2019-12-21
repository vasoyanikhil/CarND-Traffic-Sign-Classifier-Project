# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Exploratory_visualization.jpg "Visualization"
[image4]: ./German_Traffic_sign/test_1 "Traffic Sign 1"
[image5]: ./German_Traffic_sign/test_2 "Traffic Sign 2"
[image6]: ./German_Traffic_sign/test_3 "Traffic Sign 3"
[image7]: ./German_Traffic_sign/test_4 "Traffic Sign 4"
[image8]: ./German_Traffic_sign/test_5 "Traffic Sign 5"
[image9]: ./German_Traffic_sign/test_6 "Traffic Sign 6"
[image10]: ./German_Traffic_sign/test_7 "Traffic Sign 7"
[image11]: ./German_Traffic_sign/test_8 "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is simple visualization.we can see below image. 

![alt text][image1]

after you can see each unique class Vs total number of images all three set(validation set,training set,test set) graph 

cell no 5 in Traffic_Sign_Classifier.ipynb file.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to shuffle my X_train, y_train. Then, I used Normalization as one of the preprocessing technique. In which, the dataset (X_train, X_test, X_valid) is fed into the normalization(x_label) function which converts all the data and returns the normalized one.

why I used normalization becuase of the I visualize some images are poor contrast due to glare.

First RGB images to divided by 255 so i got result for exponetial form as i calculated validation accuracy are not coming as expected after that I changed step images are divided by 255 why divided by 255 becuase RGB images min range 0 to 255 so I divided by 255 after that thoes result to multiply by 0.8 becuase i got result in 0.00xxxx value thoes value after i add 0.1 in thoes result so I got result 1.xxxx result so all images pixel value I am getting 1.00xxx  becuase i chack all pixel value then I got normalization function.

So i am changing intensity of the pixel value all images are normal means arroud 1.00xxx like that.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I changed in Letnet aechitecture so My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6    				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flatten               | outputs 400                        			|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43  									|
| Softmax				| Outputs cross_entropy  						|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used: EPOCHS = 20, BATCH_SIZE = 128, rate = 0.001, mu = 0, sigma = 0.1. I used the same LeNet model architecture which consists of two convolutional layers and three fully connected layers. The input is an image of size (32x32x3) and output is 43 i.e. the total number of distinct classes. In the middle, I used RELU activation function after each convolutional layer as well as the first two fully connected layers. Flatten is used to convert the output of 2nd convolutional layer after pooling i.e. 5x5x16 into 400. Pooling is also done in between after the 1st and the 2nd convolutional layer. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5
* validation set accuracy of 97.9 
* test set accuracy of 90.6

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  First architecture having input 32 X 32 X 1 and output given logits 10  but after I changed architecture having input 32 X 32 X 3 and  output given logits 43.
  
* What were some problems with the initial architecture?
  Initial architecture does not have input for images becuase images having 3 channel.
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  My model training set accuracy more compare to validation set accuracy so architecture would be due to overfitting.

* Which parameters were tuned? How were they adjusted and why?
  I tuned a EPOCH and mu and sigma value.I changed sigma value and EPOCH. becuase In convolution,sigma in the Gaussian filter is to control the variation around its mean value. So as the Sigma becomes larger the more variance allowed around mean and as the Sigma becomes smaller the less variance allowed around mean.
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

As we discussed before Letnet 5 architecture I used but only i changed width of the channel.becuse I increse the sigma value so i get more clear image becuase in convolution low frequency detected.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
that 
I choosen Letnet 5 architecture choosen.
traffic sign image like only related to signal so it is low size image. we can resize like 32 X 32 X 3 channel. 
based on result are very near by validation set accuracy and training accuracy so we can say model working fine.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are Eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image10]
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image11]

The Third and seven image might be difficult to classify becuase model was not train to proper or images are not clear due to blur.may be some issue in predition label also.

and I face problem color are not identify becuase of some images are blue and some images are light blue due to thoes reason also we are failed for classsification.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Keep right  			| Keep right									|
| Yield					| Yield											|
| Stop Sign      		| Stop sign   									| 
| No entry      		| No entry   									|
| 70 km/h	      		| Turn right ahead 				 				|
| Road work  			| Road work          							|
| Pedestrians  			| Speed limit (70km/h) 							|

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75.00%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| 1.0     				| Keep right    								|
| 1.0					| Yield											|
| 1.0.	      			| Stop sign 					 				|
| 1.0				    | No entry          							|
| 0.0				    | 70 km/h           							|
| 1.0				    | Road work           							|
| 0.0				    | Pedestrians          							|

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



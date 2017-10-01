#**Traffic Sign Recognition** 

##Tunde Oladimeji Writeup

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./test_images/test_0.png "Traffic Sign 1"
[image5]: ./test_images/test_1.jpg "Traffic Sign 2"
[image6]: ./test_images/test_2.jpg "Traffic Sign 3"
[image7]: ./test_images/test_3.jpg "Traffic Sign 4"
[image8]: ./test_images/test_4.jpg "Traffic Sign 5"

## Project Traffic Sign Classifier Writeup 


---

You're reading my Writeup for my project and here is a link to my [project_code](https://github.com/toonday/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the representation of classes/labels in the training data set

![alt text][image1]

---

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

1.1 Grayscale
As a first step, I decided to convert the images to grayscale because I thought that color did not matter in classifying the images to a particular class. I assumed that the patterns in a grayscaled image should be enough to clearly differentiate each class apart. However, I decided to remove this step since I was getting lower accuracy when I grayscaled the image which informed me that my assumption was wrong.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

1.2 Normalization
Another step I experimented with was normalization of the dataset.
I decided to normalize the data by subtracting the mean and dividing by the variance.
By doing this I believe no one dimension would unneccessarily impact the learning being done by our model

1.3 Data Augmentation
I considered this step as one the next steps if I did not get the accuracy I desired.
By looking at the distribution of classes in the trainning set, we can see that some classes have way more distribtuion than other classes.
This might make the model better at classifying those classes than other classes that have less data in the distribution.
I decided not do this since I was able to get the accuracy required by the excercise.
To Augment the data I would have considered using techniques such as random noise, adding blurs, modifying the brightness, changing the rotation and modifying the original image dimensions.

####2. Describe what your final model architecture looks like.

My final model was a LeNet model consisting of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		|												|
| RELU					|												|
| Fully connected		|												|
| RELU					|												|
| Fully connected		|												|
| RELU					|												|
| Dropout				|												|
| Fully connected		|												|
|						|												|

####3. Describe how you trained your model.

To train the model, I did the following:
* Used an initial constant learning rate of 0.005
* Applied an exponential decay to the learning rate to reduce the learning rate by 71.5% after 10 steps
* Used a batch size of 256 (I experimented with other batch sizes, but a batch size of 256 worked best for me. The larger the batch size the worse the model performed.)
* Trained the model for 100 epochs
* Used the Adam optimizer as the optimization algorithm

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My general approach to finding a solution was iterative.
It can be summarized as follows:
* Choose an initial model architecture
* Run model and get results
* Compare training accuracy and validation accuracy
* Decide whether overfitting or underfitting is occuring
* If underfitting occurs find ways to improve the model (It shows little learning is taking place :{)
* If overfitting occurs find ways to improve the data being used in training, or modify some parameters or add some layers to avoid the overfitting problem
* Choose one variable/parameter to adjust at a time, so you can understand the effect that parameter has on learning
* Start experiments with little epochs between 5 to 15, so you can quickly run experiments, only increase epoch size when you are getting closer to your target accuracy
* Repeat step from the beginning

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.930 
* test set accuracy of 0.908

More information:
* What was the first architecture that was tried and why was it chosen?
    The first architecture I decide to try was the LeNet architecture. I chose it because it was quick and simple to develop.
* What were some problems with the initial architecture?
    I did not notice much problem with the architecture, It was a good choice in this scenario. However, there were some challenges in parameter tuning and getting the model to perform better. I would like to experiment with more recent architectures and get a better intuition of how they perform against a simple LeNet architecture. The inception module in particular seems very interesting. My guess is that it would help with overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    I tried multiple experiments, to me there was no point moving up to a more complex architecure if a simple one performs well on the problem. The only reasons I would have done that were if I could not achieve the stated performance targets or just for pure intellectual curiosity.
* Which parameters were tuned? How were they adjusted and why?
    I experimented with dropout layers to reduce overfitting.
    I got good results when I added a single dropout layer after the activation layer of the first convolution layer.
    I got poor results when I added multiple dropout layers after several activation layers (I did not understand why).
    The model performed best when I kept a single dropout layer after the activation layer of the fully connected layers.
    I did not perfrom more experiments with modifying the model architecure specifically because I was getting improved performance by modifying other parameters such as learning rate, epoch size, etc (Also I am under a huge time constraint :{)
* What are some of the important design choices and why were they chosen?
    This is a good question.
    How do folks decide what to design for?
    My design decision was guided by simplicity. I wanted to keep things simple so I understand what is going on and only add layer of complexity if I am not getting the desired accuracy.
    This means reusing as much as I could from the LeNet Architecture.
    I could experiment with increasing the depth of the convolution layers and notice how that affects the accuracy. (My guess is that the model should perform better).
    Convolution Layers work here because the problem is a pattern recognition problem. By using convolution layers, our model can learn simple patterns from the images then more complex patterns as we add more convolution layers.
    Dropout Layers could work since they would let our model to ignore some learned weights during training and force the model re-learn those weights. This should help with having the model overfit the training data set. If the weight is truly important, it would get learnt again by the model during training

Architecture description:
* What architecture was chosen?
    I chose the LeNet architecture
* Why did you believe it would be relevant to the traffic sign application?
    The problem was simple enough for this model to work for it.
    It felt like a little more complex than the pattern recognition problem of recognizing digits and letters.
    So a bit more complex patterns to recognize, LeNet was up to the Challenge! :}
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    First of all the accuracies were all high which is a great place to start ;}
    Since the accuracy on the training and validation data where high, we can assume that our model is not underfitting or overfitting the data. Furthermore, since the accuracy was also high on the test set, we can be quite confident that we have a good model to run on more problems. The word evidence feels like a strong word, I only see evidence when the model performs great on the problem itself (As in actual scenarios)

---

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Right Ahead 		| Turn Right Ahead								| 
| Yield     			| Yield 										|
| Stop					| No passing for vehicles over 3.5 metric tons	|
| 30 km/h	      		| Road work  					 				|
| Keep Right			| Keep Right         							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares poorly to the accuracy on the test set of ~91%.

From some experiments I ran the model performs better when given clean images of traffic signs with no surrounding noise in the background, but performs poorly when there are other objects or items in the surroundind scene of the traffic sign (e.g water marks, other buildings, etc)

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

For the first image, the model is extremely sure that this is a Turn Right Ahead sign (probability of 1.0), and the image does contain a Turn Right Ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn Right Ahead   							| 
| 0.00     				| Ahead only 									|
| 0.00					| Beware of ice/snow 							|
| 0.00	      			| Turn left ahead   							|
| 0.00				    | Speed limit (30km/h)         					|

For the second image, the model is extremely sure that this is a Yield sign (probability of 1.0), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield   	 									| 
| 0.00     				| Priority road 								|
| 0.00					| Speed limit (70km/h) 							|
| 0.00	      			| Speed limit (30km/h) 							|
| 0.00				    | Speed limit (50km/h) 							|

For the third image, the model is very certain that this is a No passing for vehicles over 3.5 metric tons sign (probability of 0.99), unfortunately the image does contain a No passing for vehicles over 3.5 metric tons sign. The right prediction should have been a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| No passing for vehicles over 3.5 metric tons	| 
| 0.00     				| Priority road 								|
| 0.00					| Speed limit (80km/h) 	 						|
| 0.00	      			| Speed limit (60km/h) 							|
| 0.00				    | Road work 									|


For the fourth image, the model is very certain that this is a Speed limit (120km/h) sign (probability of 0.92), unfortunately the image does contain a Speed limit (120km/h) sign. The right prediction should have been a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.92         			| Speed limit (120km/h)							| 
| 0.02         			| Speed limit (20km/h)							|
| 0.02				    | Vehicles over 3.5 metric tons prohibited 		| 
| 0.01         			| Speed limit (60km/h)							|
| 0.01	      			| Speed limit (100km/h) 						|

For the fifth image, the model is extremely sure that this is a Keep right sign (probability of 1.0), and the image does contain a Keep right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right   	 								| 
| 0.00					| Speed limit (20km/h) 							|
| 0.00					| Speed limit (30km/h) 							|
| 0.00	      			| Speed limit (50km/h) 							|
| 0.00				    | Speed limit (60km/h) 							|


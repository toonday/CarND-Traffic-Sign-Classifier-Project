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
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## 
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading my Writeup for my project and here is a link to my [project_code](https://github.com/toonday/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

[NOT ANSWERED]
A good exploratory visualization point can be to explore how many classes I have represented in my data set.
Which might show my model having a bias of identifying certain classes and not others
####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I thought that color did not matter in matching the images to a particular class. I assumed that the patterns in a grayscaled image should be enough to clearly differentiate each class apart. However, I decided to remove this step since I was getting lower accuracy when I grayscaled the image which informed me that my assumption was wrong.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

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
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I did the following:
* Used an initial learning rate of 0.005
* Applied an exponential decay to the learning rate to reduce the learning rate by 70% after 10 steps
* Used a batch size of 128 (I experimented with other batch sizes, but a batch size of 128 worked best for me. The larger the batch size the worse the model performed.)
* Trained the model for 40 epochs
* Used the Adam optimizer as the optimization algorithm

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My general approach to finding a solution was iterative.
It can be summarized as follows:
* Choose an initial model architecture
* Run model and get results
* Compare training accuracy and validation accuracy
* Decide whether overfitting or underfitting is occuring
* If underfitting occurs find ways to improve the model (It shows little learning is taking place :{)
* If overfitting occurs find ways to improve the data being used in training
* Choose one variable to tune or one experiment to run/test to get better results

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.930 
* test set accuracy of 0.911

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    The first architecture I decide to try was the LeNet architecture. I chose it because it was quick and simple to develop.
* What were some problems with the initial architecture?
    I did not notice much problem with the architecture, It was a good choice in this scenario. However, there were some challenges in parameter tuning and getting the model to perform better. I would like to experiment with more recent architectures and get a better intuition of how they perform against a simple LeNet architecture. The inception module in particular seems very interesting. My guess is that it would help with overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    I tried multiple experiments, to me there was no point moving up to a more complex architecure if a simple one performs well on the problem. The only reasons I would have done that were if I could not achieve the stated performance targets or just for pure intellectual curiosity.
* Which parameters were tuned? How were they adjusted and why?
    I added dropout layers, but I did not get the improvements I was hoping for :{
    The model performed worse when I added dropouts at the end of multiple layers, but performed in a simailar fashion when I only had the dropout at the end of the first convolution layer only.
    Since I could not notice any dramatic improvements with the dropout layer(s) I decided to comment them out
    I did not perfrom more experiments with modifying the model architecure specifically because I was getting improved performance by modifying other parameters such as learning rate, epoch size, etc (Also I am under a huge time constraint :{)
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
    I chose the LeNet architecture
* Why did you believe it would be relevant to the traffic sign application?
    The problem was simple enough for this model to work for it.
    It felt like a little more complex than the pattern recognition problem of recognizing digits and letters.
    So a bit more complex patterns to recognize, LeNet was up to the Challenge! :}
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    First of all the accuracies were all high which is a great place to start ;}
    Since the accuracy on the training and validation data where high, we can assume that our model is not underfitting or overfitting the data. Furthermore, since the accuracy was also high on the test set, we can be quite confident that we have a good model to run on more problems. The word evidence feels like a strong word, I am only see evidence when the model performs great on the problem itself (As in actual scenarios)
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



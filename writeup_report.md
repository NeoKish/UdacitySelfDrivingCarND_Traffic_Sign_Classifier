# Traffic Sign Classifier 

This project [Project] is part of Udacity Self Driving Car NanoDegree project. In this project, we were exposed to concepts of Neural Networks like softmax,cross entropy, gradient descent optimizer,regularization techiques like dropout, pooling, types of Neural Networks such as Convolutional Neural Network and the famous LeNet model Architecture. For this project, we are using open source Tensorflow library  for building and training the models and are using the course provided workspace enabled with GPU. 

The goal of the project is to build, train, validate and test a deep learning model on German Traffic images dataset. The requirement of project is to achieve more than 93% validation accuracy and then test it with new set of 5 images obtained from web to gauge the accuracy of the model. There is no requirement to perform at any percentage on these new set of images but it is a good way to understand how the model reacts to a different set of test images. 

[//]: # (File References)

[Project]: ./Traffic_Sign_Classifier.html
[Folder1]: ./data
[Image1]: ./output_images/y_train_plot_hist.png
[Folder2]: ./german_web_images

## Step 0 : Load the Data

In the first step we load the pickled data from [Folder1]. The data is already separated into train, test and validation data. In case this was not available we could have used train test split function available from scikit packages. 

The pickled data is a dictionary with 4 key/value pairs:

'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES

We separate out the features(images) and labels(sign_names) sections and save it into X and y arrays for all sets of data. 

## Step 1 : Dataset Summary and Exploration

In the second step, we used python function like len() to find out the number of samples in each dataset and function shape to find out the height and width of the images. To find out the unique classes we have used numpy function unique.To gauge the types and number of image, we did a histogram visualisation of labels for training dataset. The visualisation image can be found on [Image1] . As evidentfrom the image, we have 43 unique classes and the dataset contains more of speed limit images.


## Step 2 : Design and Test a Model Architecture

In the third step, we train and test the model.The images are normalized before feeding it to training model so that the data has zero mean and equal variance. The formula used for normalization is (image_data - 128)/ 128. I did try to use grayscale but I didn't achieve much improvement with it. For training the model, I have followed the Lenet model architecture which is described below

### Input

The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels which is 3 in our case.

### Architecture

Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
Activation. Rectifed Linear Unit(ReLu) is used 
Pooling. Input = 28x28x6. Output = 14x14x6.
Layer 2: Convolutional. Output = 10x10x16.
Activation. Rectifed Linear Unit(ReLu) is used 
Pooling. Input = 10x10x16. Output = 5x5x16.
Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. Input = 5x5x16. Output = 400.
Layer 3: Fully Connected. Input = 400. Output = 120.
Activation. Rectifed Linear Unit(ReLu) is used .
Layer 4: Fully Connected. Input = 120. Output = 84.
Activation. Rectifed Linear Unit(ReLu) is used .
Layer 5: Fully Connected. Input = 84. Output = 43.

### Output

Return the result of the 2nd fully connected layer.


The reason to choose this architecture was because it had been used in similar application for identifying traffic signs and the model is a good one since it has two convolutional network layers with max poolimg applied for each layer which helps in identifying important features from the images.

I did try adding dropout technique but it caused drop in the validation accuracy. I am guessing reason is since the dataset isn't that large, the dropout techniques results to loss of important features from the learning process.

I experimented with different value of hyperparameters such as optimizer, batch size , number of epochs and learning rate

With optimizer, I tested with Adam and Gradient descent. Out of two Adam was much better in accuracy.
With batch size, I tried with 32, 64, 128 . The batch size of 64 and 32 had similar accuracy results but 64 was faster.
With learning rate, I tried with 0.0001, 0.005 and 0.001. 0.001 had better results.
With number of epochs, I achieved constant validation accuracy values after 130 but had kept to 150 to show that it has reached constant.

My final model results were:
* validation set accuracy of 94.1
* test set accuracy of 93.4


## Step 3 : Test a Model on New Images


We have selected five images from the web to gauge the performance of the model . The images are stored in [Folder 2]

The images were normalized before testing with model. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road work    			| Road work 									|
| 30 km/h				| Keep right									|
| No entry	      		| No entry  					 				| 
| Yield     			| No passing for vehicles over 3.5 metric tons  |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

The code for calculating softmax probabilities are present in the next cell of notebook

For the images of stop sign ,road work and no entry, the model is prefectly sure (probability of 1) while for other images of yield and 30km/hr the correct prediction probabilities are way low. One way to look at the wrong predictions could be that the images were resized and it caused pixelation of images. Maybe if we try with images that are not highly pixelated then it could present better results.







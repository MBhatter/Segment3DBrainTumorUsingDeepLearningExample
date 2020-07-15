%% template for image segementation
%% instantiate imagesegmenation class
a = ImageSegmentationBaseClass

%% load your data from a csv file and run any preprocessing
a.preprocess('mydata.csv')

%% load the neural network structure

%% train the model on the training set for each fold in the k-fold

%% evaluate the average dice similarity

%% compare to MONSTR

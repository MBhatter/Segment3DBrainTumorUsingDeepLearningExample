%% 3-D Brain Tumor Segmentation Using Deep Learning
% This example shows how to train a 3-D U-Net neural network and perform semantic 
% segmentation of brain tumors from 3-D medical images.
% 
% The example shows how to train a 3-D U-Net network and also provides a pretrained 
% network. If you choose to train the network, use of a CUDA-capable NVIDIA™ GPU 
% with compute capability 3.0 or higher is highly recommended (requires Parallel 
% Computing Toolbox™).
%% Introduction
% Semantic segmentation involves labeling each pixel in an image or voxel of 
% a 3-D volume with a class.
% 
% This example illustrates the use of deep learning methods to semantically 
% segment brain tumors in magnetic resonance imaging (MRI) scans. It also shows 
% how to perform binary segmentation, in which each voxel is labeled as tumor 
% or background. 
% 
% One challenge of medical image segmentation is class imbalance in the data 
% that hampers training when using conventional cross entropy loss. This example 
% solves the problem by using a weighted multiclass Dice loss function [4]. Weighing 
% the classes helps to counter the influence of larger regions on the Dice score, 
% making it easier for the network to learn how to segment smaller regions.
% 
% This example performs brain tumor segmentation using a 3-D U-Net architecture 
% [1]. U-Net is a fast, efficient and simple network that has become popular in 
% the semantic segmentation domain.
%% Download Training, Validation, and Test Data
% This example uses the BraTS data set [2]. The BraTS data set contains MRI 
% scans of brain tumors, namely gliomas, which are the most common primary brain 
% malignancies. The size of the data file is ~7 GB. If you do not want to download 
% the BraTS data set, then go directly to the _Download Pretrained Network and 
% Sample Test Set_ section in this example.
%% 
% Create a directory to store the BraTS data set.

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
%% 
% To download the BraTS data, go to the <http://medicaldecathlon.com/ Medical 
% Segmentation Decathalon> website and click the "Download Data" link. Download 
% the "Task01_BrainTumour.tar" file [3]. Unzip the TAR file into the directory 
% specified by the |imageDir| variable. When unzipped successfully, |imageDir| 
% will contain a directory named |Task01_BrainTumour| that has three subdirectories: 
% |imagesTr|, |imagesTs|, and |labelsTr|. 
% 
% The data set contains 750 4-D volumes, each representing a stack of 3-D images. 
% Each 4-D volume has size 240-by-240-by-155-by-4, where the first three dimensions 
% correspond to height, width, and depth of a 3-D volumetric image. The fourth 
% dimension corresponds to different scan modalities. Each image volume has a 
% corresponding pixel label. The data set is divided into 484 training volumes 
% and 286 test volumes.
%% Preprocess Training and Validation Data
% To train the 3-D U-Net network more efficiently, preprocess the MRI data using 
% the helper function |preprocessBraTSdataset|. This function is attached to the 
% example as a supporting file.
% 
% The helper function performs these operations:
%% 
% * Crop the data to a region containing primarily the brain and tumor. Cropping 
% the data reduces the size of data while retaining the most critical part of 
% each MRI volume and its corresponding labels.
% * Normalize each modality of each volume independently by subtracting the 
% mean and dividing by the standard deviation of the cropped brain region.
% * Split the data set into training, validation, and test sets.
%% 
% Preprocessing the data can take about 30 minutes to complete. 

sourceDataLoc = [imageDir filesep 'Task01_BrainTumour'];
preprocessDataLoc = fullfile(tempdir,'BraTS','preprocessedDataset');
preprocessBraTSdataset(preprocessDataLoc,sourceDataLoc);
%% Create Random Patch Extraction Datastore for Training and Validation
% Use a random patch extraction datastore to feed the training data to the network 
% and to validate the training progress. This datastore extracts random patches 
% from ground truth images and corresponding pixel label data. Patching is a common 
% technique to prevent running out of memory when training with arbitrarily large 
% volumes
% 
% First, store the training images in an <docid:matlab_ref#butueui-1 |imageDatastore|>. 
% Because the MAT-file format is a nonstandard image format, you must use a MAT-file 
% reader to enable reading the image data. You can use the helper MAT-file reader, 
% |matRead|. This function is attached to the example as a supporting file. 

volReader = @(x) matRead(x);
volLoc = fullfile(preprocessDataLoc,'imagesTr');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);
%% 
% Create a <docid:vision_ref#mw_c2246553-ba4a-4bad-aad4-6ab8fa2f7f2d |pixelLabelDatastore|> 
% to store the labels.

labelReader = @(x) matRead(x);
lblLoc = fullfile(preprocessDataLoc,'labelsTr');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);
%% 
% Preview one image volume and label. Display the labeled volume using the <docid:images_ref#mw_7a40592d-db0e-4bdb-9ba3-446bd1715151 
% |labelvolshow|> function. Make the background fully transparent by setting the 
% visibility of the background label (|1|) to |0|.

volume = preview(volds);
label = preview(pxds);
figure
h = labelvolshow(label,volume(:,:,:,1));
h.LabelVisibility(1) = 0;
%% 
% Create a <docid:images_ref#mw_19a16ac8-a068-411c-8f32-def517a4399a |randomPatchExtractionDatastore|> 
% from the image datastore and pixel label datastore. Specify a patch size of 
% 64-by-64-by-64 voxels. Specify |'PatchesPerImage'| to extract 16 randomly positioned 
% patches from each pair of volumes and labels during training. Specify a mini-batch 
% size of 8.

patchSize = [64 64 64];
patchPerImage = 16;
miniBatchSize = 8;
patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;
%% 
% Augment the training data by using the <docid:matlab_ref#mw_16489124-fe7e-4381-b715-8d3b8b30a9f6 
% |transform|> function with custom preprocessing operations specified by the 
% helper function |augment3dPatch|. The |augment3dPatch| function randomly rotates 
% and reflects the training data to make the training more robust. This function 
% is attached to the example as a supporting file.

dsTrain = transform(patchds,@augment3dPatch);
%% 
% Create a |randomPatchExtrationDatastore| that contains the validation data. 
% You can use validation data to evaluate whether the network is continuously 
% learning, underfitting, or overfitting as time progresses.

volLocVal = fullfile(preprocessDataLoc,'imagesVal');
voldsVal = imageDatastore(volLocVal, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblLocVal = fullfile(preprocessDataLoc,'labelsVal');
pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;
%% Set Up 3-D U-Net Layers
% This example uses a variation of the 3-D U-Net network [1]. In U-Net, the 
% initial series of convolutional layers are interspersed with max pooling layers, 
% successively decreasing the resolution of the input image. These layers are 
% followed by a series of convolutional layers interspersed with upsampling operators, 
% successively increasing the resolution of the input image. The name U-Net comes 
% from the fact that the network can be drawn with a symmetric shape like the 
% letter U. This example modifies the U-Net to use zero-padding in the convolutions, 
% so that the input and the output to the convolutions have the same size.
% 
% This example defines the 3-D U-Net using layers from Deep Learning Toolbox™, 
% including:
%% 
% * <docid:nnet_ref#object_layer.imageinput3dlayer |imageInput3dLayer|>* *- 
% 3-D image input layer
% * <docid:nnet_ref#object_layer.convolution3dlayer |convolution3dLayer|>* *- 
% 3-D convolution layer for convolutional neural networks
% * <docid:nnet_ref#mw_b7913af4-3a40-4020-bb2c-18c946f5eadd |batchNormalizationLayer|> 
% - Batch normalization layer
% * <docid:nnet_ref#mw_ca5427bd-5cdc-4a58-ba63-302c257d8222 |reluLayer|>* *- 
% Leaky rectified linear unit layer
% * <docid:nnet_ref#object_layer.maxpooling3dlayer |maxPooling3dLayer|> - 3-D 
% max pooling layer
% * <docid:nnet_ref#function_transposedconv3dlayer |transposedConv3dLayer|>* 
% *- Transposed 3-D convolution layer
% * <docid:nnet_ref#mw_a09d3c68-d062-4692-a950-9a7fea5c40c3 |softmaxLayer|>* 
% *- Softmax output layer
% * <docid:nnet_ref#object_layer.concatenationlayer |concatenationLayer|> - 
% Concatenation layer
%% 
% This example also defines a custom Dice loss layer, named |dicePixelClassification3dLayer|, 
% to solve the problem of class imbalance in the data [4]. This layer is attached 
% to the example as a supporting file. For more information, see <docid:nnet_examples.mw_010f8e56-c48d-4f8b-9657-a046780d7f6e 
% Define Custom Pixel Classification Layer with Dice Loss>. 
%% 
% The first layer, |imageInput3dLayer|, operates on image patches of size 64-by-64-by-64 
% voxels.

inputSize = [64 64 64 4];
inputLayer = image3dInputLayer(inputSize,'Normalization','none','Name','input');
%% 
% The image input layer is followed by the contracting path of the 3-D U-Net. 
% The contracting path consists of three encoder modules. Each encoder contains 
% two convolution layers with 3-by-3-by-3 filters that double the number of feature 
% maps, followed by a nonlinear activation using reLu layer. The first convolution 
% is also followed by a batch normalization layer. Each encoder ends with a max 
% pooling layer that halves the image resolution in each dimension.
% 
% Give unique names to all the layers. The layers in the encoder have names 
% starting with "en" followed by the index of the encoder module. For example, 
% "en1" denotes the first encoder module. 

numFiltersEncoder = [
    32 64; 
    64 128; 
    128 256];

layers = [inputLayer];
for module = 1:3
    modtag = num2str(module);
    encoderModule = [
        convolution3dLayer(3,numFiltersEncoder(module,1), ...
            'Padding','same','WeightsInitializer','narrow-normal', ...
            'Name',['en',modtag,'_conv1']);
        batchNormalizationLayer('Name',['en',modtag,'_bn']);
        reluLayer('Name',['en',modtag,'_relu1']);
        convolution3dLayer(3,numFiltersEncoder(module,2), ...
            'Padding','same','WeightsInitializer','narrow-normal', ...
            'Name',['en',modtag,'_conv2']);
        reluLayer('Name',['en',modtag,'_relu2']);
        maxPooling3dLayer(2,'Stride',2,'Padding','same', ...
            'Name',['en',modtag,'_maxpool']);
    ];
    
    layers = [layers; encoderModule];
end
%% 
% Create the expanding path of the 3-D U-Net. The expanding path consists of 
% four decoder modules. All decoders contain two convolution layers with 3-by-3-by-3 
% filters that halve the number of feature maps, followed by a nonlinear activation 
% using a reLu layer. The first three decoders conclude with a transposed convolution 
% layer that upsamples the image by a factor of 2. The final decoder includes 
% a convolution layer that maps the feature vector of each voxel to the classes.
% 
% Give unique names to all the layers. The layers in the decoder have names 
% starting with "de" followed by the index of the decoder module. For example, 
% "de4" denotes the fourth decoder module.

numFiltersDecoder = [
    256 512; 
    256 256; 
    128 128; 
    64 64];

decoderModule4 = [
    convolution3dLayer(3,numFiltersDecoder(1,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de4_conv1');
    reluLayer('Name','de4_relu1');
    convolution3dLayer(3,numFiltersDecoder(1,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de4_conv2');
    reluLayer('Name','de4_relu2');
    transposedConv3dLayer(2,numFiltersDecoder(1,2),'Stride',2, ...
        'Name','de4_transconv');
];

decoderModule3 = [
    convolution3dLayer(3,numFiltersDecoder(2,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ....
        'Name','de3_conv1');       
    reluLayer('Name','de3_relu1');
    convolution3dLayer(3,numFiltersDecoder(2,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de3_conv2'); 
    reluLayer('Name','de3_relu2');
    transposedConv3dLayer(2,numFiltersDecoder(2,2),'Stride',2, ...
        'Name','de3_transconv');
];

decoderModule2 = [
    convolution3dLayer(3,numFiltersDecoder(3,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de2_conv1');
    reluLayer('Name','de2_relu1');
    convolution3dLayer(3,numFiltersDecoder(3,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de2_conv2');
    reluLayer('Name','de2_relu2');
    transposedConv3dLayer(2,numFiltersDecoder(3,2),'Stride',2, ...
        'Name','de2_transconv');
];
%% 
% The final decoder includes a convolution layer that maps the feature vector 
% of each voxel to each of the two classes (tumor and background). The custom 
% Dice pixel classification layer weights the loss function to increase the impact 
% of the small tumor regions on the Dice score.

numLabels = 2;
decoderModuleFinal = [
    convolution3dLayer(3,numFiltersDecoder(4,1), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de1_conv1');
    reluLayer('Name','de1_relu1');
    convolution3dLayer(3,numFiltersDecoder(4,2), ...
        'Padding','same','WeightsInitializer','narrow-normal', ...
        'Name','de1_conv2');
    reluLayer('Name','de1_relu2');
    convolution3dLayer(1,numLabels,'Name','convLast');
    softmaxLayer('Name','softmax');
    dicePixelClassification3dLayer('output');
];
%% 
% Concatenate the input layer and encoder modules with the fourth decoder module. 
% Add the other decoder modules to the layer graph as separate branches.

layers = [layers; decoderModule4];
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph,decoderModule3);
lgraph = addLayers(lgraph,decoderModule2);
lgraph = addLayers(lgraph,decoderModuleFinal);
%% 
% % 
% Use concatenation layers to connect the second reLu layer of each encoder 
% module with a transposed convolution layer of equal size from a decoder module. 
% Connect the output of each concatenation layer to the first convolution layer 
% of the decoder module.

concat1 = concatenationLayer(4,2,'Name','concat1');
lgraph = addLayers(lgraph,concat1);
lgraph = connectLayers(lgraph,'en1_relu2','concat1/in1');
lgraph = connectLayers(lgraph,'de2_transconv','concat1/in2');
lgraph = connectLayers(lgraph,'concat1/out','de1_conv1');

concat2 = concatenationLayer(4,2,'Name','concat2');
lgraph = addLayers(lgraph,concat2);
lgraph = connectLayers(lgraph,'en2_relu2','concat2/in1');
lgraph = connectLayers(lgraph,'de3_transconv','concat2/in2');
lgraph = connectLayers(lgraph,'concat2/out','de2_conv1');

concat3 = concatenationLayer(4,2,'Name','concat3');
lgraph = addLayers(lgraph,concat3);
lgraph = connectLayers(lgraph,'en3_relu2','concat3/in1');
lgraph = connectLayers(lgraph,'de4_transconv','concat3/in2');
lgraph = connectLayers(lgraph,'concat3/out','de3_conv1'); 
%% 
% % 
% Alternatively, you can use the |createUnet3d| helper function to create the 
% 3-D U-Net layers. This function is attached to the example as a supporting file.

lgraph = createUnet3d(inputSize);
%% 
% Plot the layer graph.

analyzeNetwork(lgraph)
%% Specify Training Options
% Train the network using the "adam" optimization solver. Specify the hyperparameter 
% settings using the <docid:nnet_ref#bu59f0q |trainingOptions|> function. The 
% initial learning rate is set to 5e-4 and gradually decreases over the span of 
% training.

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);
%% Download Pretrained Network and Sample Test Set
% Optionally, download a pretrained version of 3-D U-Net and five sample test 
% volumes and their corresponding labels from the BraTS data set [3]. The pretrained 
% model and sample data enable you to perform segmentation on test data without 
% downloading the full data set or waiting for the network to train.

trained3DUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNet.mat';
sampleData_url = 'https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSet.tar.gz';

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
downloadTrained3DUnetSampleData(trained3DUnet_url,sampleData_url,imageDir);
%% Train the Network
% After configuring the training options and the data source, train the 3-D 
% U-Net network by using the <docid:nnet_ref#bu6sn4c |trainNetwork|> function. 
% To train the network, set the |doTraining| parameter in the following code to 
% |true|. A CUDA-capable NVIDIA™ GPU with compute capability 3.0 or higher is 
% highly recommended for training.
% 
% If you keep the |doTraining| parameter in the following code as |false|, then 
% the example returns a pretrained 3-D U-Net network.
% 
% _Note: Training takes about 60 hours on an NVIDIA™ Titan X and can take even 
% longer depending on your GPU hardware._

doTraining = false;
if doTraining
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net');
else
    load(fullfile(imageDir,'trained3DUNet','brainTumor3DUNet.mat'));
end
%% 
% % 
% You can now use the U-Net to semantically segment brain tumors.
%% Perform Segmentation of Test Data
% Select the source of test data that contains ground truth volumes and labels 
% for testing. If you keep the |useFullTestSet| parameter in the following code 
% as |false|, then the example uses five volumes for testing. If you set the |useFullTestSet| 
% parameter to |true|, then the example uses 55 test images selected from the 
% full data set.

useFullTestSet = false;
if useFullTestSet
    volLocTest = fullfile(preprocessDataLoc,'imagesTest');
    lblLocTest = fullfile(preprocessDataLoc,'labelsTest');
else
    volLocTest = fullfile(imageDir,'sampleBraTSTestSet','imagesTest');
    lblLocTest = fullfile(imageDir,'sampleBraTSTestSet','labelsTest');
    classNames = ["background","tumor"];
    pixelLabelID = [0 1];
end
%% 
% Crop the central portion of the images and labels to size 128-by-128-by-128 
% voxels by using the helper function |centerCropMatReader|. This function is 
% attached to the example as a supporting file. The |voldsTest| variable stores 
% the ground truth test images. The |pxdsTest| variable stores the ground truth 
% labels.

windowSize = [128 128 128];
volReader = @(x) centerCropMatReader(x,windowSize);
labelReader = @(x) centerCropMatReader(x,windowSize);
voldsTest = imageDatastore(volLocTest, ...
    'FileExtensions','.mat','ReadFcn',volReader);
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);
%% 
% For each test image, add the ground truth image volumes and labels to cell 
% arrays. Use the trained network with the <docid:vision_ref#mw_bbecb1af-a6c9-43d1-91f5-48607edc15d1 
% |semanticseg|> function to predict the labels for each test volume.
% 
% After performing the segmentation, postprocess the predicted labels by labeling 
% nonbrain voxels as |1|, corresponding to the background. Use the test images 
% to determine which voxels do not belong to the brain. You can also clean up 
% the predicted labels by removing islands and filling holes using the <docid:images_ref#bvb_85o-1 
% |medfilt3|> function. |medfilt3| does not support categorical data, so cast 
% the pixel label IDs to |uint8| before the calculation. Then, cast the filtered 
% labels back to the categorical data type, specifying the original pixel label 
% IDs and class names.

id=1;
while hasdata(voldsTest)
    disp(['Processing test volume ' num2str(id)])
    
    groundTruthLabels{id} = read(pxdsTest);
    
    vol{id} = read(voldsTest);
    tempSeg = semanticseg(vol{id},net);

    % Get the non-brain region mask from the test image.
    volMask = vol{id}(:,:,:,1)==0;
    % Set the non-brain region of the predicted label as background.
    tempSeg(volMask) = classNames(1);
    % Perform median filtering on the predicted label.
    tempSeg = medfilt3(uint8(tempSeg)-1);
    % Cast the filtered label to categorial.
    tempSeg = categorical(tempSeg,pixelLabelID,classNames);
    predictedLabels{id} = tempSeg;
    id=id+1;
end
%% Compare Ground Truth Against Network Prediction
% Select one of the test images to evaluate the accuracy of the semantic segmentation. 
% Extract the first modality from the 4-D volumetric data and store this 3-D volume 
% in the variable |vol3d|.

volId = 2;
vol3d = vol{volId}(:,:,:,1);
%% 
% Display in a montage the center slice of the ground truth and predicted labels 
% along the depth direction.

zID = size(vol3d,3)/2;
zSliceGT = labeloverlay(vol3d(:,:,zID),groundTruthLabels{volId}(:,:,zID));
zSlicePred = labeloverlay(vol3d(:,:,zID),predictedLabels{volId}(:,:,zID));

figure
title('Labeled Ground Truth (Left) vs. Network Prediction (Right)')
montage({zSliceGT;zSlicePred},'Size',[1 2],'BorderSize',5) 
%% 
% Display the ground-truth labeled volume using the <docid:images_ref#mw_7a40592d-db0e-4bdb-9ba3-446bd1715151 
% |labelvolshow|> function. Make the background fully transparent by setting the 
% visibility of the background label (|1|) to |0|. Because the tumor is inside 
% the brain tissue, make some of the brain voxels transparent, so that the tumor 
% is visible. To make some brain voxels transparent, specify the volume threshold 
% as a number in the range [0, 1]. All normalized volume intensities below this 
% threshold value are fully transparent. This example sets the volume threshold 
% as less than 1 so that some brain pixels remain visible, to give context to 
% the spatial location of the tumor inside the brain.

figure
h1 = labelvolshow(groundTruthLabels{volId},vol3d);
h1.LabelVisibility(1) = 0;
h1.VolumeThreshold = 0.68;
%% 
% For the same volume, display the predicted labels.

figure
h2 = labelvolshow(predictedLabels{volId},vol3d);
h2.LabelVisibility(1) = 0;
h2.VolumeThreshold = 0.68;
%% 
% This image shows the result of displaying slices sequentially across the entire 
% volume.
% 
% %% Quantify Segmentation Accuracy
% Measure the segmentation accuracy using the <docid:images_ref#mw_1ee709d7-bf6b-4ac9-8f5d-e7caf72497d4 
% |dice|> function. This function computes the Dice similarity coefficient between 
% the predicted and ground truth segmentations.

diceResult = zeros(length(voldsTest.Files),2);

for j = 1:length(vol)
    diceResult(j,:) = dice(groundTruthLabels{j},predictedLabels{j});
end
%% 
% Calculate the average Dice score across the set of test volumes.

meanDiceBackground = mean(diceResult(:,1));
disp(['Average Dice score of background across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceBackground)])
meanDiceTumor = mean(diceResult(:,2));
disp(['Average Dice score of tumor across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceTumor)])
%% 
% The figure shows a <docid:stats_ug#bu180jd |boxplot|> that visualizes statistics 
% about the Dice scores across the set of five sample test volumes. The red lines 
% in the plot show the median Dice value for the classes. The upper and lower 
% bounds of the blue box indicate the 25th and 75th percentiles, respectively. 
% Black whiskers extend to the most extreme data points not considered outliers.
% 
% % 
% If you have Statistics and Machine Learning Toolbox™, then you can use the 
% |boxplot| function to visualize statistics about the Dice scores across all 
% your test volumes. To create a |boxplot|, set the |createBoxplot| parameter 
% in the following code to |true|.

createBoxplot = false;
if createBoxplot
    figure
    boxplot(diceResult)
    title('Test Set Dice Accuracy')
    xticklabels(classNames)
    ylabel('Dice Coefficient')
end
%% Summary
% This example shows how to create and train a 3-D U-Net network to perform 
% 3-D brain tumor segmentation using the BraTS data set. The steps to train the 
% network include:
%% 
% * Download and preprocess the training data.
% * Create a <docid:images_ref#mw_19a16ac8-a068-411c-8f32-def517a4399a |randomPatchExtractionDatastore|> 
% that feeds training data to the network. 
% * Define the layers of the 3-D U-Net network.
% * Specify training options.
% * Train the network using the <docid:nnet_ref#bu6sn4c |trainNetwork|> function.
%% 
% After training the 3-D U-Net network or loading a pretrained 3-D U-Net network, 
% the example performs semantic segmentation of a test data set. The example evaluates 
% the predicted segmentation by a visual comparison to the ground truth segmentation 
% and by measuring the Dice similarity coefficient between the predicted and ground 
% truth segmentation.
%% References
% [1] Çiçek, Ö., A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger. 
% "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." In 
% _Proceedings of the International Conference on Medical Image Computing and 
% Computer-Assisted Intervention_. Athens, Greece, Oct. 2016, pp. 424-432.
% 
% [2] Isensee, F., P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein. 
% "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to 
% the BRATS 2017 Challenge." In _Proceedings of BrainLes: International MICCAI 
% Brainlesion Workshop_. Quebec City, Canada, Sept. 2017, pp. 287-297.
% 
% [3] "Brain Tumours". _Medical Segmentation Decathalon._ http://medicaldecathlon.com/ 
% 
% The BraTS dataset is provided by Medical Decathlon under the <https://creativecommons.org/licenses/by-sa/4.0/ 
% CC-BY-SA 4.0 license.> All warranties and representations are disclaimed; see 
% the license for details. MathWorks® has modified the data set linked in the 
% _Download Pretrained Network and Sample Test Set_ section of this example. The 
% modified sample dataset has been cropped to a region containing primarily the 
% brain and tumor and each channel has been normalized independently by subtracting 
% the mean and dividing by the standard deviation of the cropped brain region.
% 
% [4] Sudre, C. H., W. Li, T. Vercauteren, S. Ourselin, and M. J. Cardoso. "Generalised 
% Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations." 
% _Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical 
% Decision Support: Third International Workshop_. Quebec City, Canada, Sept. 
% 2017, pp. 240-248.
% 
% _Copyright 2018 The MathWorks, Inc._
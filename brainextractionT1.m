%% 3-D Brain Extraction from MRI
%    Train and cross validate a 3-D U-net for brain extraction on T1 image
%% load nifti data 
%   manu -
%          load nifti data from /rsrch1/ip/egates1/NFBS\ Skull\ Strip/NFBSFilepaths.csv 
%          into matlab data structure

%% setup data for k-fold cross validation
%   aurian - split the data into training/ validation/  test  sets


%% load the 3D U-net structure
%  priya - 

% before starting, need to define "n" which is the number of channels.
n = 4;
lgraph = layerGraph();

tempLayers = [
    image3dInputLayer([64 64 64 n],"Name","input","Normalization","none")
    convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    reluLayer("Name","relu_Module1_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module2_Level1")
    reluLayer("Name","relu_Module2_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module3_Level1")
    reluLayer("Name","relu_Module3_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
    convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")
    convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")
    transposedConv3dLayer([2 2 2],512,"Name","transConv_Module4","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat3")
    convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")
    transposedConv3dLayer([2 2 2],256,"Name","transConv_Module5","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat2")
    convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")
    transposedConv3dLayer([2 2 2],128,"Name","transConv_Module6","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")
    convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassification3dLayer("output")];
    
%     helperDicePixelClassification3dLayer("output",1e-08,categorical(["background";"tumor"]));

lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in1");
lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in2");
lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in2");
lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in2");

plot(lgraph);

%% train the model on the training set for each fold in the k-fold

%% evaluate the average dice similarity
%   manu - output nifti files of the test set predictions
%   priya - visualize differences in itksnap

%% compare to MONSTR
%   @egates1 - do you have MONSTR predictions on this dataset ? are there any papers that have already done this ?

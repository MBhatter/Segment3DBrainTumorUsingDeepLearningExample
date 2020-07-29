%% liver segmentation on MRI
%% liver segmentation on MRI
% Setting up the code: fresh start
clear all
close all

%% instantiate class
% TODO - load  multiple networks ? 

% https://www.mathworks.com/help/matlab/matlab_external/use-python-dict-type-in-matlab.html
% order = py.dict(pyargs('soup',3.57,'bread',2.29,'bacon',3.91,'salad',5.00))

% I ran this code only for the 1st file in the trainingList, but I should
% be able to do a for loop and run the rest of the files by changing fname
% for each file

trainingList = { './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/000/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/001/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/002/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/003/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/004/setup.json'};
fname = '/rsrch1/ip/dtfuentes/github/Segment3DBrainTumorUsingDeepLearningExample/hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/000/setup.json';
jsonText = fileread(fname);
jsonData = jsondecode(jsonText);


% Read file pathways into table
fullFileName = jsonData.fullFileName;

delimiter = jsonData.delimiter;
    % enter: ,

T = readtable(fullFileName, 'Delimiter', delimiter);
A = table2cell(T);
volCol = jsonData.volCol;
    % enter: 4
    
lblCol = jsonData.lblCol;
    % enter: 5

volLoc = A(:,volCol);
lblLoc = A(:,lblCol);

stoFoldername = jsonData.stoFoldername;
destination = fullfile(tempdir,stoFoldername, 'preprocessedDataset');

outputdir = '/rsrch1/ip/dtfuentes/github/Segment3DBrainTumorUsingDeepLearningExample/hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/000';

% define readers
 maskReader = @(x) (niftiread(x)>0);
 volReader = @(x) niftiread(x);
 
 %read data into datastores
 % ERRORS: for some reason the filenames became anonymized but fortunately
 % I already finished the preprocessing for the 1st file in trainingList
 % before this change
 volds = imageDatastore(volLoc, ...
     'FileExtensions','.gz','ReadFcn',volReader);
 
 classNames = ["background","brain"];
  pixelLabelID = [0 1];
 
 % read data intp pixelLabeldatastore
 pxds = pixelLabelDatastore(lblLoc,classNames, pixelLabelID, ...
        'FileExtensions','.gz','ReadFcn',maskReader);
  reset(volds);
  reset(pxds);      
 

   % create directories to store data sets
        mkdir(fullfile(destination,'imagesMain'));
        mkdir(fullfile(destination,'labelsMain'));
        
        imDir = fullfile(destination, 'imagesMain', stoFoldername);
        labelDir = fullfile(destination, 'labelsMain', stoFoldername);
    

       
   %% Crop relevant region
    NumFiles = length(pxds.Files);
    id = 1;
    
    while hasdata(pxds)
        outL = readNumeric(pxds);
        outV = read(volds);
        temp = outL>0;
        sz = size(outL);
        reg = regionprops3(temp,'BoundingBox');
        tol = 64;
        ROI = ceil(reg.BoundingBox(1,:));
        ROIst = ROI(1:3) - tol;
        ROIend = ROI(1:3) + ROI(4:6) + tol;

        ROIst(ROIst<1)=1;
        ROIend(ROIend>sz)=sz(ROIend>sz);

        tumorRows = ROIst(2):ROIend(2);
        tumorCols = ROIst(1):ROIend(1);
        tumorPlanes = ROIst(3):ROIend(3);

        tcropVol = outV(tumorRows,tumorCols, tumorPlanes);
        tcropLabel = outL(tumorRows,tumorCols, tumorPlanes);


% Data set with a valid size for 3-D U-Net (multiple of 8)
        ind = floor(size(tcropVol)/8)*8;
        incropVol = tcropVol(1:ind(1),1:ind(2),1:ind(3));
        mask = incropVol == 0;

%%%%%%%% channelWisePreProcess
        % As input has 4 channels (modalities), remove the mean and divide by the
        % standard deviation of each modality independently.
        incropVol1=single(incropVol);

        chn_Mean = mean(incropVol1,[1 2 3]);
        chn_Std = std(incropVol1,0,[1 2 3]);
        cropVol1 = (incropVol1 - chn_Mean)./chn_Std;

        rangeMin = -5;
        rangeMax = 5;
        % Remove outliers
        cropVol1(cropVol1 > rangeMax) = rangeMax;
        cropVol1(cropVol1 < rangeMin) = rangeMin;

        % Rescale the data to the range [0, 1]
        cropVol1 = (cropVol1 - rangeMin) / (rangeMax - rangeMin);

%%%%%%%%        
        % Set the nonbrain region to 0
        cropVol1(mask) = 0;
        cropLabel = tcropLabel(1:ind(1),1:ind(2),1:ind(3));

        
        % save preprocessed data to folders
         save([imDir num2str(id,'%.3d') '.mat'],'cropVol1');
         save([labelDir num2str(id,'%.3d') '.mat'],'cropLabel');
         
         %outDim{id} = size(cropVol1);
        
         id=id+1;

   end  
 
   
   

%% make directories for 2d slices
mkdir(fullfile(outputdir, '2dimages'));
mkdir(fullfile(outputdir, '2dlabels'));

mkdir(fullfile(outputdir, '2d validation images'));
mkdir(fullfile(outputdir, '2d validation labels'));

% directories for training images 
trimfolderPath = fullfile(outputdir, '2dimages'); %'/rsrch1/ip/dtfuentes/github/Segment3DBrainTumorUsingDeepLearningExample/hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/000/2dimages';
trfolderlblpath = fullfile(outputdir, '2dlabels'); %'/rsrch1/ip/dtfuentes/github/Segment3DBrainTumorUsingDeepLearningExample/hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/000/2dlabels';

valimfolderPath = fullfile(outputdir, '2d validation images');
valLblfolderpath = fullfile(outputdir, '2d validation labels');

%% test set: make loop to read in all the .mat files and convert to 2d slices

imDirmain = fullfile(destination, 'imagesMain');
labelDirmain = fullfile(destination, 'labelsMain');
trainset = jsonData.trainset;
numFiles = length(trainset);


for id = 1:numFiles
    voldest = fullfile(imDirmain, trainset{id});
    imLoad = load(voldest);
    imVol = imLoad.cropVol1;
    format2file(imVol, trimfolderPath, id);
    
    lbldest = fullfile(labelDirmain, trainset{id});
    lblLoad = load(lbldest);
    lblVol = lblLoad.cropLabel;
    labels2file(lblVol, trfolderlblpath, id);
end


%% validation set: make loop to read through validation data
valset = jsonData.validationset;
numFiles = length(valset);


for id = 1:numFiles
    voldest2 = fullfile(imDirmain, valset{id});
    imLoad2 = load(voldest2);
    imVol2 = imLoad2.cropVol1;
    format2file(imVol2, valimfolderPath, id);
    
    lbldest2 = fullfile(labelDirmain, valset{id});
    lblLoad2 = load(lbldest2);
    lblVol2 = lblLoad2.cropLabel;
    labels2file(lblVol2, valLblfolderpath, id);
end

%% store images in imagedastore and pixellabeldatastore

% training datastores
trimds = imageDatastore(trimfolderPath);
classnames = ["background" "liver"];
pixelLabelID = [0 1];
trlblds = pixelLabelDatastore(trfolderlblpath, classnames, pixelLabelID);

% validation datastores
valimds = imageDatastore(valimfolderPath);
vallblds = pixelLabelDatastore(valLblfolderpath, classnames, pixelLabelID);


% Need Random Patch Extraction on testing and validation Data
patchSize = [64 64];
patchPerImage = 4;
miniBatchSize = 8;
  %training patch datastore
trpatchds = randomPatchExtractionDatastore(trimds,trlblds,patchSize, ...
    'PatchesPerImage',patchPerImage);
trpatchds.MiniBatchSize = miniBatchSize;
  %validation patch datastore
dsVal = randomPatchExtractionDatastore(valimds,vallblds,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;


minibatch = preview(trpatchds);
montage(minibatch.InputImage)

%% load network
%specify the n as the number of channels
n = 1;
% Create Layer Graph
% Create the layer graph variable to contain the network layers.

lgraph = layerGraph();
% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    imageInputLayer([64 64 n],"Name","input","Normalization","none")
    convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    reluLayer("Name","relu_Module1_Level1")
    convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module2_Level1")
    reluLayer("Name","relu_Module2_Level1")
    convolution2dLayer([3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module3_Level1")
    reluLayer("Name","relu_Module3_Level1")
    convolution2dLayer([3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")
    convolution2dLayer([3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")
    transposedConv2dLayer([2 2],512,"Name","transConv_Module4","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat3")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")
    transposedConv2dLayer([2 2],256,"Name","transConv_Module5","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat2")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")
    transposedConv2dLayer([2 2],128,"Name","transConv_Module6","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat1")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")
    convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassificationLayer('Name', 'output')];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in2");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in2");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in2");
lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in1");
lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in1");
lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in1");
% Plot Layers

plot(lgraph);


%% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false);

doTraining = true;
if doTraining 
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(trpatchds,lgraph,options);
    save(['trained2DUNet-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');
    infotable = struct2table(info);
    writetable(infotable, ['2DUNetinfo-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.txt']);
end

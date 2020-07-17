%% liver segmentation on MRI
% Setting up the code: fresh start
clear all
close all

%% instantiate class
% TODO - load  multiple networks ? 

% https://www.mathworks.com/help/matlab/matlab_external/use-python-dict-type-in-matlab.html
% order = py.dict(pyargs('soup',3.57,'bread',2.29,'bacon',3.91,'salad',5.00))

% walker - can we avoid the if statment here a load a class based on the json file ? 
a = hccmriunet3d ('./hccmrilog/dscimg/densenet2d/adadelta/512/run_a/005020/005/000/setup.json') ; 
b = hccmriunet2d ('./hccmrilog/dscimg/densenet2d/adadelta/512/run_a/005020/005/000/setup.json') ; 


%% load nifti data 
%a.preprocess()

% before starting, need to define "n" which is the number of channels.
NumberOfChannels = 1;
a.loadneuralnet(NumberOfChannels)

% functiom point to load mat files
procReader = @(x) matRead(x);

% read image volume data
trainData      = imageDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/imagesMain/',a.jsonData.trainset     ) , 'FileExtensions','.mat','ReadFcn',procReader);
validationData = imageDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/imagesMain/',a.jsonData.validationset) , 'FileExtensions','.mat','ReadFcn',procReader);

% read label data
fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/labelsMain/',a.jsonData.trainset)
    
% read these into pixellabeldatastores
classNames = ["background","liver"];
pixelLabelID = [0 1];
trainMask      = pixelLabelDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/labelsMain/',a.jsonData.trainset     ),classNames,pixelLabelID, 'FileExtensions','.mat','ReadFcn',procReader );
validationMask = pixelLabelDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/labelsMain/',a.jsonData.validationset),classNames,pixelLabelID, 'FileExtensions','.mat','ReadFcn',procReader );

% Need Random Patch Extraction on testing and validation Data
patchSize = [64 64 64];
patchPerImage = 16;
miniBatchSize = 8;
  %training patch datastore
trpatchds = randomPatchExtractionDatastore(trainData,trmaskpxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
trpatchds.MiniBatchSize = miniBatchSize;
  %validation patch datastore
dsVal = randomPatchExtractionDatastore(valData,valmaskpxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;

% training options
options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);
    
    
% train and save 
modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
[net,info] = trainNetwork(trpatchds,a.lgraph,options);
save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');


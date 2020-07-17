%% liver segmentation on MRI
% Setting up the code: fresh start
clear all
close all

%% instantiate class
% TODO - load  multiple networks ? 

% https://www.mathworks.com/help/matlab/matlab_external/use-python-dict-type-in-matlab.html
% order = py.dict(pyargs('soup',3.57,'bread',2.29,'bacon',3.91,'salad',5.00))

trainingList = { './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/000/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/001/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/002/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/003/setup.json', './hccmrilog/dscimg/unet2d/adadelta/512/run_a/005020/005/004/setup.json'};

for idtrain= 1: numel(trainingList )
  % walker - best way to parallelize ? 
  a = hccmriunet2d (trainingList{idtrain} ) ; 
  
  %% load nifti data 
  a.preprocess()
  
  % before starting, need to define "n" which is the number of channels.
  NumberOfChannels = 1;
  a.loadneuralnet(NumberOfChannels)
  
  % functiom point to load mat files
  procReader = @(x) matRead(x);
  
  % read image volume data
  trainData      = imageDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/imagesMain/',a.jsonData.trainset     ) , 'FileExtensions','.mat','ReadFcn',procReader);
  validationData = imageDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/imagesMain/',a.jsonData.validationset) , 'FileExtensions','.mat','ReadFcn',procReader);
  
  % read these into pixellabeldatastores
  classNames = ["background","liver"];
  pixelLabelID = [0 1];
  trainMask      = pixelLabelDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/labelsMain/',a.jsonData.trainset     ),classNames,pixelLabelID, 'FileExtensions','.mat','ReadFcn',procReader );
  validationMask = pixelLabelDatastore(fullfile(a.jsonData.stoFoldername,'/preprocessedDataset/labelsMain/',a.jsonData.validationset),classNames,pixelLabelID, 'FileExtensions','.mat','ReadFcn',procReader );
  
  % Need Random Patch Extraction on testing and validation Data
  miniBatchSize = 8;
  %training datastore
  trainingSet = pixelLabelImageDatastore(trainData,trainMask,patchSize, 'MiniBatchSize', miniBatchSize);
  %validation datastore
  validationSet = pixelLabelImageDatastore(validationData,validationMask, 'MiniBatchSize', miniBatchSize);
  
  % training options
  options = trainingOptions('adam', ...
      'MaxEpochs',50, ...
      'InitialLearnRate',5e-4, ...
      'LearnRateSchedule','piecewise', ...
      'LearnRateDropPeriod',5, ...
      'LearnRateDropFactor',0.95, ...
      'ValidationData',validationSet, ...
      'ValidationFrequency',400, ...
      'Plots','training-progress', ...
      'Verbose',false, ...
      'MiniBatchSize',miniBatchSize);
      
  % train and save 
  modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS')
  [net,info] = trainNetwork(trainingSet,a.lgraph,options);
  save([a.jsonData.uidoutputdir '/trained3DUNet-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');

end

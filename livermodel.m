%% liver segmentation on MRI
% Setting up the code: fresh start
clear all
close all

%% instantiate class
a = ImageSegmentationBaseClass ('./hccmrilog/dscimg/densenet2d/adadelta/512/run_a/005020/005/000/setup.json') ; 
a = ImageSegmentationUnet2D ('./hccmrilog/dscimg/densenet2d/adadelta/512/run_a/005020/005/000/setup.json') ; 

a = livermodelclass ('liverConfig.json') ; 

%% load nifti data 
[procvolds, proclblfs] =  a.preprocess()

% before starting, need to define "n" which is the number of channels.
NumberOfChannels = 1;
a.LoadNNUnet3d(NumberOfChannels)
% TODO - @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
% TODO - load  multiple networks
% a.LoadNNDUnet2d(NumberChannels)

% split the data into k-folds
patients  = table2array(a.tabledb(:,1));% Extract the patient ids in the filepaths table
partition = cvpartition(patients,'k',5);

% TODO - @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
% TODO - loop over multiple the NN model for each k-fold
% train the model on the training set for each fold in the k-fold
% and write out the trained model
for i = 1:partition.NumTestSets
    trIdx = partition.training(i);
    teIdx = partition.test(i);
    trData = subset(procvolds, trIdx);
    trMask = subset(proclblfs, trIdx);

    % Training, validation, and test data for each fold
    % TODO - @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
    %  what is 12 and 13 ? 
    trainData = subset(trData, [13:.8*length(patients)]);
    trainMask = subset(trMask, [13:.8*length(patients)]);
    valData = subset(trData, [1:12]);
    valMask = subset(trMask, [1:12]);
    testData = subset(procvolds, teIdx);
    testMask = subset(proclblfs, teIdx);
    
    % write file pathways of mask sets from dsfileset to table
    trmaskinfo = resolve(trainMask);
    valmaskinfo = resolve(valMask);
    testmaskinfo = resolve(testMask);
    
    % convert tables to arrays
    trmaskfullArr = table2array(trmaskinfo);
    valmaskfullArr = table2array(valmaskinfo);
    testmaskfullArr = table2array(testmaskinfo);
    
    % read file pathways into string arrays
    trmaskArr = trmaskfullArr(:,1);
    valmaskArr = valmaskfullArr(:,1);
    testmaskArr = testmaskfullArr(:,1);
    
    % convert string to char arrays
    trmaskChar = convertStringsToChars(trmaskArr);
    valmaskChar = convertStringsToChars(valmaskArr);
    testmaskChar = convertStringsToChars(testmaskArr);
    
    % read these into pixellabeldatastores
    proclblReader = @(x) matRead(x);
    classNames = ["background","tumor"];
    pixelLabelID = [0 1];
    trmaskpxds = pixelLabelDatastore(trmaskChar,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',proclblReader );

    valmaskpxds = pixelLabelDatastore(valmaskChar,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',proclblReader );

    tsmaskpxds = pixelLabelDatastore(testmaskChar,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',proclblReader );

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
    
    
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(trpatchds,a.lgraph,options);
    save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');

end



%% evaluate the average dice similarity
%   manu - output nifti files of the test set predictions
%   priya - visualize differences in itksnap

%% compare to MONSTR
%   @egates1 - do you have MONSTR predictions on this dataset ? are there any papers that have already done this ?

%% 3-D Brain Extraction from MRI
%    Train and cross validate a 3-D U-net for brain extraction on T1 image
%% load nifti data 
%   manu -
%          load nifti data from /rsrch1/ip/egates1/NFBS\ Skull\ Strip/NFBSFilepaths.csv 
%          into matlab data structure
% Setting up the code: fresh start
clear all
close all

a = livermodelclass 
a.preprocess()


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %% Do the k-fold partition
%% 
%% patients = tabledb(:,1);% Extract the patient ids in the filepaths table
%% partition = cvpartition(patients,'k',5);
%% err = zeros(partition.NumTestSets,1);
%% 
%% 
%% for i = 1:partition.NumTestSets
%%     trIdx = partition.training(i);
%%     teIdx = partition.test(i);
%%     trData = subset(procvolds, trIdx);
%%     trMask = subset(proclblfs, trIdx);
%% 
%%     % Training, validation, and test data for each fold
%%     trainData = subset(trData, [13:.8*length(patients)]);
%%     trainMask = subset(trMask, [13:.8*length(patients)]);
%%     valData = subset(trData, [1:12]);
%%     valMask = subset(trMask, [1:12]);
%%     testData = subset(procvolds, teIdx);
%%     testMask = subset(proclblfs, teIdx);
%%     
%%     % write file pathways of mask sets from dsfileset to table
%%     trmaskinfo = resolve(trainMask);
%%     valmaskinfo = resolve(valMask);
%%     testmaskinfo = resolve(testMask);
%%     
%%     % convert tables to arrays
%%     trmaskfullArr = table2array(trmaskinfo);
%%     valmaskfullArr = table2array(valmaskinfo);
%%     testmaskfullArr = table2array(testmaskinfo);
%%     
%%     % read file pathways into string arrays
%%     trmaskArr = trmaskfullArr(:,1);
%%     valmaskArr = valmaskfullArr(:,1);
%%     testmaskArr = testmaskfullArr(:,1);
%%     
%%     % convert string to char arrays
%%     trmaskChar = convertStringsToChars(trmaskArr);
%%     valmaskChar = convertStringsToChars(valmaskArr);
%%     testmaskChar = convertStringsToChars(testmaskArr);
%%     
%%     % read these into pixellabeldatastores
%%     proclblReader = @(x) matRead(x);
%%     classNames = ["background","tumor"];
%%     pixelLabelID = [0 1];
%%     trmaskpxds = pixelLabelDatastore(trmaskChar,classNames,pixelLabelID, ...
%%     'FileExtensions','.mat','ReadFcn',procvolReader);
%% 
%%     valmaskpxds = pixelLabelDatastore(valmaskChar,classNames,pixelLabelID, ...
%%     'FileExtensions','.mat','ReadFcn',procvolReader);
%% 
%%     tsmaskpxds = pixelLabelDatastore(testmaskChar,classNames,pixelLabelID, ...
%%     'FileExtensions','.mat','ReadFcn',procvolReader);
%% 
%% 
%%     % Compute Dice(general concept, might be a more code-friendly way to do it)
%%     %{
%%     p = networkPrediction.*correctPrediction
%%     s = 2*sum(p, 'all')
%%     err(i) = s/(sum(networkPrediction,'all')+sum(correctPrediction, 'all'))
%%     %}
%% 
%% end
%% % Average Loss Function Error for all folds
%% %cvErr = sum(err)/sum(partition.TestSize);
%% 
%% 
%% % Need Random Patch Extraction on testing and validation Data
%% patchSize = [64 64 64];
%% patchPerImage = 16;
%% miniBatchSize = 8;
%%   %training patch datastore
%% trpatchds = randomPatchExtractionDatastore(trainData,trmaskpxds,patchSize, ...
%%     'PatchesPerImage',patchPerImage);
%% trpatchds.MiniBatchSize = miniBatchSize;
%%   %validation patch datastore
%% dsVal = randomPatchExtractionDatastore(valData,valmaskpxds,patchSize, ...
%%     'PatchesPerImage',patchPerImage);
%% dsVal.MiniBatchSize = miniBatchSize;
%% 
%% 
%% % before starting, need to define "n" which is the number of channels.
%% n = 1;
%% lgraph = layerGraph();
%% 
%% tempLayers = [
%%     image3dInputLayer([64 64 64 n],"Name","input","Normalization","none")
%%     convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
%%     batchNormalizationLayer("Name","BN_Module1_Level1")
%%     reluLayer("Name","relu_Module1_Level1")
%%     convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module1_Level2")];
%% lgraph = addLayers(lgraph,tempLayers);
%% tempLayers = [
%%     maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
%%     convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
%%     batchNormalizationLayer("Name","BN_Module2_Level1")
%%     reluLayer("Name","relu_Module2_Level1")
%%     convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module2_Level2")];
%% lgraph = addLayers(lgraph,tempLayers);
%% tempLayers = [
%%     maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
%%     convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
%%     batchNormalizationLayer("Name","BN_Module3_Level1")
%%     reluLayer("Name","relu_Module3_Level1")
%%     convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module3_Level2")];
%% lgraph = addLayers(lgraph,tempLayers);
%% tempLayers = [
%%     maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
%%     convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module4_Level1")
%%     convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module4_Level2")
%%     transposedConv3dLayer([2 2 2],512,"Name","transConv_Module4","Stride",[2 2 2])];
%% lgraph = addLayers(lgraph,tempLayers);
%% tempLayers = [
%%     concatenationLayer(4,2,"Name","concat3")
%%     convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module5_Level1")
%%     convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module5_Level2")
%%     transposedConv3dLayer([2 2 2],256,"Name","transConv_Module5","Stride",[2 2 2])];
%% lgraph = addLayers(lgraph,tempLayers);
%% tempLayers = [
%%     concatenationLayer(4,2,"Name","concat2")
%%     convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module6_Level1")
%%     convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
%%     reluLayer("Name","relu_Module6_Level2")
%%     transposedConv3dLayer([2 2 2],128,"Name","transConv_Module6","Stride",[2 2 2])];
%% lgraph = addLayers(lgraph,tempLayers);
%% tempLayers = [
%%     concatenationLayer(4,2,"Name","concat1")
%%     convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
%%     reluLayer("Name","relu_Module7_Level1")
%%     convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
%%     reluLayer("Name","relu_Module7_Level2")
%%     convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
%%     softmaxLayer("Name","softmax")
%%     dicePixelClassification3dLayer("output")];
%% 
%% %     helperDicePixelClassification3dLayer("output",1e-08,categorical(["background";"tumor"]));
%% 
%% lgraph = addLayers(lgraph,tempLayers);
%% 
%% % clean up helper variable
%% clear tempLayers;
%% lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
%% lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in1");
%% lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
%% lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in1");
%% lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
%% lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in1");
%% lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in2");
%% lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in2");
%% lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in2");
%% 
%% plot(lgraph);
%% 
%% %% train the model on the training set for each fold in the k-fold
%% % Need to Train the network using training and validation data
%% 
%% options = trainingOptions('adam', ...
%%     'MaxEpochs',50, ...
%%     'InitialLearnRate',5e-4, ...
%%     'LearnRateSchedule','piecewise', ...
%%     'LearnRateDropPeriod',5, ...
%%     'LearnRateDropFactor',0.95, ...
%%     'ValidationData',dsVal, ...
%%     'ValidationFrequency',400, ...
%%     'Plots','training-progress', ...
%%     'Verbose',false, ...
%%     'MiniBatchSize',miniBatchSize);
%% 
%% 
%% modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
%% [net,info] = trainNetwork(trpatchds,lgraph,options);
%% save(['trained3DUNet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net');
%% 
%% %% evaluate the average dice similarity
%% %% train the model on the training set for each fold in the k-fold
%% 
%% %% evaluate the average dice similarity
%% %   manu - output nifti files of the test set predictions
%% %   priya - visualize differences in itksnap
%% 
%% %% compare to MONSTR
%% %   @egates1 - do you have MONSTR predictions on this dataset ? are there any papers that have already done this ?

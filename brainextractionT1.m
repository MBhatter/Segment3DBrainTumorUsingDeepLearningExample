%% 3-D Brain Extraction from MRI
%    Train and cross validate a 3-D U-net for brain extraction on T1 image
%% load nifti data 
%   manu -
%          load nifti data from /rsrch1/ip/egates1/NFBS\ Skull\ Strip/NFBSFilepaths.csv 
%          into matlab data structure
% Setting up the code: fresh start
clc
clear all
close all

% Read file pathways into table
folder = '/rsrch1/ip/egates1/NFBS Skull Strip/';
fullFileName = fullfile(folder, 'NFBSFilepaths.csv');
T = readtable(fullFileName, 'Delimiter', ',');
A = table2array(T);
T1Loc = A(:,4);
maskLoc = A(:,5);

 % define readers
 maskReader = @(x) (niftiread(x)>0);
 volReader = @(x) niftiread(x);
 
 %read data into datastores
 volds = imageDatastore(T1Loc, ...
     'FileExtensions','.gz','ReadFcn',volReader);
 
 classNames = ["background","brain"];
  pixelLabelID = [0 1];
 
 % read data intp pixelLabeldatastore
 pxds = pixelLabelDatastore(maskLoc,classNames, pixelLabelID, ...
        'FileExtensions','.gz','ReadFcn',maskReader);
  reset(volds);
  reset(pxds);      

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
        
        tcropVol = outV(tumorRows,tumorCols, tumorPlanes,:);
        tcropLabel = outL(tumorRows,tumorCols, tumorPlanes);
        
        
% Data set with a valid size for 3-D U-Net (multiple of 8)
        ind = floor(size(tcropVol)/8)*8;
        incropVol = tcropVol(1:ind(1),1:ind(2),1:ind(3),:);
        mask = incropVol == 0;
        
%%%%%%%% channelWisePreProcess
        % As input has 4 channels (modalities), remove the mean and divide by the
        % standard deviation of each modality independently.
        incropVol1=single(incropVol);
        
        chn_Mean = mean(incropVol1,[1 2 3]);
        chn_Std = std(incropVol1,0,[1 2 3]);
        cropVol = (incropVol1 - chn_Mean)./chn_Std;

        rangeMin = -5;
        rangeMax = 5;
        % Remove outliers
        cropVol(cropVol > rangeMax) = rangeMax;
        cropVol(cropVol < rangeMin) = rangeMin;

        % Rescale the data to the range [0, 1]
        cropVol = (cropVol - rangeMin) / (rangeMax - rangeMin);
        
%%%%%%%%        
        % Set the nonbrain region to 0
        cropVol(mask) = 0;
        cropLabel = tcropLabel(1:ind(1),1:ind(2),1:ind(3));
        
%         % Split data into training, validation and test sets. Roughly 82%
%         % are training, 6% are validation, and 12% are test
%         if (id < floor(0.83*NumFiles))
%             imDir    = fullfile(destination,'imagesTr','BraTS');
%             labelDir = fullfile(destination,'labelsTr','BraTS');
%         elseif (id < floor(0.89*NumFiles))
%             imDir    = fullfile(destination,'imagesVal','BraTS');
%             labelDir = fullfile(destination,'labelsVal','BraTS');
%         else
%             imDir    = fullfile(destination,'imagesTest','BraTS');
%             labelDir = fullfile(destination,'labelsTest','BraTS');
%         end
%         save([imDir num2str(id,'%.3d') '.mat'],'cropVol');
%         save([labelDir num2str(id,'%.3d') '.mat'],'cropLabel');
        id=id+1;
   
   end  



%% setup data for k-fold cross validation
%   aurian - split the data into training/ validation/  test  sets


%% load the 3D U-net structure
%   priya - insert code for the 3D Unet

%% train the model on the training set for each fold in the k-fold

%% evaluate the average dice similarity
%   manu - output nifti files of the test set predictions
%   priya - visualize differences in itksnap

%% compare to MONSTR
%   @egates1 - do you have MONSTR predictions on this dataset ? are there any papers that have already done this ?

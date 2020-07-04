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
fullFileName = input("Please enter the file pathway: ", 's')
    % enter: = /rsrch1/ip/egates1/NFBS Skull Strip/NFBSFilepaths.csv

delimiter = input("Please enter the delimiter: ", 's')
    % enter: ,

T = readtable(fullFileName, 'Delimiter', delimiter);
A = table2array(T);
volCol = input("Please enter column number for volumetric data: ")
    % enter: 4
    
lblCol = input ("Please enter the column number for mask data: ")
    % enter: 5

volLoc = A(:,volCol);
lblLoc = A(:,lblCol);

% for user-defined: destination = input("Please enter the file pathway for folder to store training, validation, and test sets: ", 's')
destination = fullfile(tempdir,'brainTest', 'preprocessedDataset');

 % define readers
 maskReader = @(x) (niftiread(x)>0);
 volReader = @(x) niftiread(x);
 
 %read data into datastores
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
        
        imDir = fullfile(destination, 'imagesMain', 'brainTest');
        labelDir = fullfile(destination, 'labelsMain', 'brainTest');
    

       
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
        cropLabel = tcropLabel(1:ind(1),1:ind(2),1:ind(3))

        
        % save preprocessed data to folders
         save([imDir num2str(id,'%.3d') '.mat'],'cropVol1');
         save([labelDir num2str(id,'%.3d') '.mat'],'cropLabel');
         id=id+1;

   end  
    

%% create datastores for processed labels and images

% procvolds stores processed T1 volumetric data
procvolReader = @(x) matRead(x);
procvolLoc = fullfile(destination,'imagesMain');
procvolds = imageDatastore(procvolLoc, ...
    'FileExtensions','.mat','ReadFcn',procvolReader);

% proclblds stores processed mask volumetric data
proclblReader = @(x) matRead(x);
proclblLoc = fullfile(destination,'labelsMain');
proclblds = imageDatastore(proclblLoc, ...
    'FileExtensions','.mat','ReadFcn',proclblReader);


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

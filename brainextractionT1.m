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
fullFileName = fullfile(folder, 'NFBSFilepaths.csv')
T = readtable(fullFileName, 'Delimiter', ',')

% convert table to cell array
A = table2array(T)

% create cell arrays to hold Volumetric data
T1RAI{125,1}=[];
maskRAI{125,1}=[];
T1{125,1}=[];
mask{125,1}=[];

% niftiread 
% loop through T1RAI column and do niftiread
for row = 1:125
    T1RAI{row,1} = niftiread(A{row,2});
end

% loop through maskRAI column and do niftiread to store volumetric data
for row = 1:125
    maskRAI{row,1} = niftiread(A{row,3});
end

% loop through T1 column and do niftiread to read vol data
for row = 1:125
    T1{row,1} = niftiread(A{row,4});
end

% loop through mask column and do niftiread to read in vol data
for row = 1:125
    mask{row,1} = niftiread(A{row,5});
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

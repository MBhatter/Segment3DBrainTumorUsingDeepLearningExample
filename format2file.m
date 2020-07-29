%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function format2file(Imgs,folderPath,instanceNumber)
% Imgs are a 3D array, each 2D slice is a training point
% Folder path points to where they are going to written to file
% Instance number tells you when to start counting. You'll probably need to
% call this function a bunch of times.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% HU Conversion (Lungs)
%Imgs = 1 + ( double(Imgs)*1.00 - 1024 )/1024; 

% Number of images
N = size(Imgs,3); 

% Resize to 128
Dat = zeros(128,128,N);
for i = 1:N
    Dat(:,:,i) = imresize(Imgs(:,:,i),[128 128]); 
end

% Clean up
% scale = 5; Dat(Dat < 0) = 0; Dat(Dat > scale) = scale; Dat = Dat/scale;

% Write the images to file as .png
num = 0; if(~isempty(instanceNumber) ), num = instanceNumber; end
for i = 0:N-1
    name = sprintf('%s/Img_%06d_PT%d.png',folderPath,i+num, instanceNumber);
    imwrite(Dat(:,:,i+1),name);
end


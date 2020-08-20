function datCell = images4d2store(imVol)

%This function slices 4d images through the axial point (3rd dimension)

% find number of slices
slices = size(imVol, 3);

%create empty cell array
datCell = {};

% find number of channels
 numChannel = size(imVol, 4);

% Resize to 128
Dat = zeros(128,128,slices, numChannel);
for i = 1:slices
    Dat(:,:,i,:) = imresize(imVol(:,:,i,:),[128 128]); 
end

% Write the images to file as .mat
num = 0;

for i = 0:slices-1
    num = num+1;
    cropVol = Dat(:,:,i+1,:);
    permuteVol = permute(cropVol, [1 2 4 3]);
    datCell{num} = permuteVol;

end

end

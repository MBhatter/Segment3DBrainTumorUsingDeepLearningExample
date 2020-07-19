% inputniftifilepath - input full path to nifti file
% mynetwork - input full path to the NN we will use
% outputpath - output path where files will be written
function applymodel( inputniftifilepath, mynetwork, outputpath )

%% load nifti file
info = niftiinfo(inputniftifilepath );
niivolume = niftiread(info);

%% load trained network
load(mynetwork );

%% apply trained network to nifti image
tempSeg = semanticseg(niivolume ,net,'ExecutionEnvironment','cpu');

%% write output to disk as a nifti file

end

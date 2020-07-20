% inputniftifilepath - input full path to nifti file
% mynetwork - input full path to the NN we will use
% outputpath - output path where files will be written
function applymodel( inputniftifilepath, mynetwork, outputpath )

disp( ['inputniftifilepath= ''',inputniftifilepath,''';']);      
disp( ['mynetwork         = ''',mynetwork         ,''';']);      
disp( ['outputpath        = ''',outputpath        ,''';']);  

%% load nifti file
info = niftiinfo(inputniftifilepath );
niivolume = niftiread(info);

%% load trained network
trainedNN = load(mynetwork )

%% apply trained network to nifti image
tStart = tic;
tempSeg = semanticseg(niivolume ,trainedNN.net,'ExecutionEnvironment','cpu');
tEnd = toc(tStart)

%% write output to disk as a nifti file
outputlabel = fullfile(outputpath,'label')
infoout = info;
infoout.Filename = outputlabel;
infoout.Datatype = 'uint8';
niftiwrite(uint8(tempSeg) ,outputlabel ,infoout,'Compressed',true)

end

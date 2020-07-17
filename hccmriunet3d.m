classdef hccmriunet3d < ImageSegmentationUnet3D  
   methods
      function preprocess(obj)
        volCol = obj.jsonData.volCol;
        lblCol = obj.jsonData.lblCol;
        
        uidlist = table2array(obj.tabledb(:,2));
        volLoc  = table2array(obj.tabledb(:,volCol));
        lblLoc  = table2array(obj.tabledb(:,lblCol));
        
        stoFoldername = obj.jsonData.stoFoldername;
        % for user-defined: destination = input("Please enter the file pathway for folder to store training, validation, and test sets: ", 's')
        destination = fullfile(stoFoldername, 'preprocessedDataset');
        
        % define readers
        maskReader = @(x) (niftiread(x)>0);
        volReader = @(x) niftiread(x);
        
        %read data into datastores
        volds = imageDatastore(volLoc, ...
            'FileExtensions','.gz','ReadFcn',volReader);
        
        classNames = ["background","liver"];
        pixelLabelID = [0 1];
        
        % read data intp pixelLabeldatastore
        pxds = pixelLabelDatastore(lblLoc,classNames, pixelLabelID, ...
               'FileExtensions','.gz','ReadFcn',maskReader);
        reset(volds);
        reset(pxds);      
        
        % create directories to store data sets
        mkdir(fullfile(destination,'imagesMain'));
        mkdir(fullfile(destination,'labelsMain'));
               
        imDir = fullfile(destination, 'imagesMain/');
        labelDir = fullfile(destination, 'labelsMain/');
              
        %% Crop relevant region
        NumFiles = length(pxds.Files);
        id = 1;
           
        while hasdata(pxds)
            % print progress
            disp([':', id,'  ',uidlist{id},' ',pxds.Files{id},' ',volds.Files{id}])
            outL = readNumeric(pxds);
            outV = read(volds);
        
            tcropVol   = outV;
            tcropLabel = outL;
        
            % Data set with a valid size for 3-D U-Net (multiple of 8)
            ind = floor(size(tcropVol)/8)*8;
            incropVol = tcropVol(1:ind(1),1:ind(2),1:ind(3));
            mask = incropVol == 0;
        
            %%%%%%%% channelWisePreProcess
            % As input has 4 channels (modalities), remove the mean and divide by the
            % standard deviation of each modality independently.
            incropVol1=single(incropVol);
        
            % TODO - @mbhatter - channel dependent here -
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
        
            % Set the nonbrain region to 0
            cropVol1(mask) = 0;
            cropLabel = tcropLabel(1:ind(1),1:ind(2),1:ind(3));
            
            % save preprocessed data to folders
            save([imDir uidlist{id}  '.mat'],'cropVol1');
            save([labelDir uidlist{id}  '.mat'],'cropLabel');
             
            %outDim{id} = size(cropVol1);
            id=id+1;
        
        end  
        
      end
   end
end


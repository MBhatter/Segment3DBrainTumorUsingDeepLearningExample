%  @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
%   base class for image segmentation

classdef ImageSegmentationBaseClass
   properties
      Value {mustBeNumeric}
      lgraph %  @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
             %  TODO - here lgraph is a scalar, can we make this a vector ? I want to load multiple networks 
      tabledb  %  
      jsonData %  
   end
   methods

      function obj = ImageSegmentationBaseClass(fname)
        % constructor - load all configuration data
        jsonText = fileread(fname);
        obj.jsonData = jsondecode(jsonText);
        
        % Read file pathways into table
        fullFileName = obj.jsonData.fullFileName;
        delimiter = obj.jsonData.delimiter;
        obj.tabledb = readtable(fullFileName, 'Delimiter', delimiter);
      end


      function preprocess(obj,filename)
         disp('overload me with your data specific preprocessing')
         disp(['load data from a  your csv file - ', filename])
      end
      % load 3d Unet, input: number of channels
      function LoadNNUnet3d(obj,NumberChannels)
         obj.lgraph = layerGraph();
         
         tempLayers = [
             image3dInputLayer([64 64 64 NumberChannels],"Name","input","Normalization","none")
             convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module1_Level1")
             reluLayer("Name","relu_Module1_Level1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module1_Level2")];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module2_Level1")
             reluLayer("Name","relu_Module2_Level1")
             convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module2_Level2")];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module3_Level1")
             reluLayer("Name","relu_Module3_Level1")
             convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module3_Level2")];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level1")
             convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level2")
             transposedConv3dLayer([2 2 2],512,"Name","transConv_Module4","Stride",[2 2 2])];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         tempLayers = [
             concatenationLayer(4,2,"Name","concat3")
             convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level1")
             convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level2")
             transposedConv3dLayer([2 2 2],256,"Name","transConv_Module5","Stride",[2 2 2])];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         tempLayers = [
             concatenationLayer(4,2,"Name","concat2")
             convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level1")
             convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level2")
             transposedConv3dLayer([2 2 2],128,"Name","transConv_Module6","Stride",[2 2 2])];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         tempLayers = [
             concatenationLayer(4,2,"Name","concat1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
             reluLayer("Name","relu_Module7_Level1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
             reluLayer("Name","relu_Module7_Level2")
             convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
             softmaxLayer("Name","softmax")
             dicePixelClassification3dLayer("output")];
         
         %     helperDicePixelClassification3dLayer("output",1e-08,categorical(["background";"tumor"]));
         
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         % clean up helper variable
         clear tempLayers;
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module1_Level2","maxpool_Module1");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module1_Level2","concat1/in1");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module2_Level2","maxpool_Module2");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module2_Level2","concat2/in1");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module3_Level2","maxpool_Module3");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module3_Level2","concat3/in1");
         obj.lgraph = connectLayers(obj.lgraph,"transConv_Module4","concat3/in2");
         obj.lgraph = connectLayers(obj.lgraph,"transConv_Module5","concat2/in2");
         obj.lgraph = connectLayers(obj.lgraph,"transConv_Module6","concat1/in2");
         
         plot(obj.lgraph);
      end
      % load 3d DenseUnet, input: number of channels
      function LoadNNDenseUnet3d(obj,NumberChannels)
         disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
      end
      % load deepmedic DenseUnet, input: number of channels
      function LoadNNDeepMedic(obj,NumberChannels)
         disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
      end
      % load 2D  , input: number of channels
      function LoadNNDenseUnet2d(obj,NumberChannels)
         disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
      end
      % load 2D  , input: number of channels
      function LoadNNDUnet2d(obj,NumberChannels)
         disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
      end
      % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR  - what other NN methods should we add ? 
   end
end


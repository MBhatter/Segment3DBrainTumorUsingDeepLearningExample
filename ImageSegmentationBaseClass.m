%  @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
%   base class for image segmentation

classdef ImageSegmentationBaseClass  < handle
   properties
      Value {mustBeNumeric}
      lgraph %  @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
             %  TODO - here lgraph is a scalar, can we make this a vector ? I want to load multiple networks 
      tabledb  %  
      jsonData % 
      netTypes
   end
   
   %%
   methods

      function obj = ImageSegmentationBaseClass(fname)
%         % constructor - load all configuration data
%         jsonText = fileread(fname);
%         obj.jsonData = jsondecode(jsonText);
%         
%         % Read file pathways into table
%         fullFileName = obj.jsonData.fullFileName;
%         delimiter = obj.jsonData.delimiter;
%         obj.tabledb = readtable(fullFileName, 'Delimiter', delimiter);\
        obj.jsonData = fname;

        % initialize NN
        % changed NumberChannels to 4 for example, and used denseUnet2d
        % as a placeholder for Deep Medic
        obj.lgraph{1} = obj.LoadNNUnet3d(4);
        obj.lgraph{2} = obj.LoadNNDenseUnet3d(4);
        obj.lgraph{3} = obj.LoadNNDenseUnet2d(4);
        obj.lgraph{4} = obj.LoadNNDenseUnet2d(4);
        obj.lgraph{5} = obj.LoadNNUnet2d(4);

        obj.netTypes = ["Unet3d","DenseUnet3d","DeepMedic","DenseUnet2d","Unet2d"];
      end

% 
%       function preprocess(obj,filename)
%          disp('overload me with your data specific preprocessing')
%          disp(['load data from a  your csv file - ', filename])
%       end
   end
   
   %%
   methods(Static)
      % load 3d Unet, input: number of channels
      function lgraph = LoadNNUnet3d(NumberChannels)
         lgraph = layerGraph();
         tempLayers = [
             image3dInputLayer([64 64 64 NumberChannels],"Name","input","Normalization","none")
             convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module1_Level1")
             reluLayer("Name","relu_Module1_Level1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module1_Level2")];
         lgraph = addLayers(lgraph,tempLayers);
         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module2_Level1")
             reluLayer("Name","relu_Module2_Level1")
             convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module2_Level2")];
         lgraph = addLayers(lgraph,tempLayers);
         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module3_Level1")
             reluLayer("Name","relu_Module3_Level1")
             convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module3_Level2")];
         lgraph = addLayers(lgraph,tempLayers);
         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level1")
             convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level2")
             transposedConv3dLayer([2 2 2],512,"Name","transConv_Module4","Stride",[2 2 2])];
         lgraph = addLayers(lgraph,tempLayers);
         tempLayers = [
             concatenationLayer(4,2,"Name","concat3")
             convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level1")
             convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level2")
             transposedConv3dLayer([2 2 2],256,"Name","transConv_Module5","Stride",[2 2 2])];
         lgraph = addLayers(lgraph,tempLayers);
         tempLayers = [
             concatenationLayer(4,2,"Name","concat2")
             convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level1")
             convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level2")
             transposedConv3dLayer([2 2 2],128,"Name","transConv_Module6","Stride",[2 2 2])];
         lgraph = addLayers(lgraph,tempLayers);
         tempLayers = [
             concatenationLayer(4,2,"Name","concat1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
             reluLayer("Name","relu_Module7_Level1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
             reluLayer("Name","relu_Module7_Level2")
             convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
             softmaxLayer("Name","softmax")
             dicePixelClassification3dLayer("output")];
                  
         lgraph = addLayers(lgraph,tempLayers);
         
         % clean up helper variable
         clear tempLayers;
         lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
         lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in1");
         lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
         lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in1");
         lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
         lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in1");
         lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in2");
         lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in2");
         lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in2");
        
         plot(lgraph);
      end
      
      %%
      % load 3d DenseUnet, input: number of channels
      function lgraph = LoadNNDenseUnet3d(NumberChannels)
         disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
         lgraph = layerGraph();

         tempLayers = [
             image3dInputLayer([64 64 64 NumberChannels],"Name","input","Normalization","none")
             convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module1_Level1")
             reluLayer("Name","relu_Module1_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module1_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = concatenationLayer(4,2,"Name","concat_1");
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module2_Level1")
             reluLayer("Name","relu_Module2_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module2_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = concatenationLayer(4,2,"Name","concat_2");
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module3_Level1")
             reluLayer("Name","relu_Module3_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module3_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = concatenationLayer(4,2,"Name","concat_3");
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
             convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat_4")
             transposedConv3dLayer([2 2 2],512,"Name","transConv_Module4","Stride",[2 2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat3")
             convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat_5")
             transposedConv3dLayer([2 2 2],256,"Name","transConv_Module5","Stride",[2 2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat2")
             convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat_6")
             transposedConv3dLayer([2 2 2],128,"Name","transConv_Module6","Stride",[2 2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat1")
             convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
             reluLayer("Name","relu_Module7_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
             reluLayer("Name","relu_Module7_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(4,2,"Name","concat")
             convolution3dLayer([1 1 1],2,"Name","ConvLast_Module7")
             softmaxLayer("Name","softmax")
             dicePixelClassification3dLayer("output")];
         lgraph = addLayers(lgraph,tempLayers);

         % clean up helper variable
         clear tempLayers;

         lgraph = connectLayers(lgraph,"relu_Module1_Level1","conv_Module1_Level2");
         lgraph = connectLayers(lgraph,"relu_Module1_Level1","concat_1/in1");
         lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat_1/in2");
         lgraph = connectLayers(lgraph,"concat_1","maxpool_Module1");
         lgraph = connectLayers(lgraph,"concat_1","concat1/in1");
         lgraph = connectLayers(lgraph,"relu_Module2_Level1","conv_Module2_Level2");
         lgraph = connectLayers(lgraph,"relu_Module2_Level1","concat_2/in1");
         lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat_2/in2");
         lgraph = connectLayers(lgraph,"concat_2","maxpool_Module2");
         lgraph = connectLayers(lgraph,"concat_2","concat2/in1");
         lgraph = connectLayers(lgraph,"relu_Module3_Level1","conv_Module3_Level2");
         lgraph = connectLayers(lgraph,"relu_Module3_Level1","concat_3/in1");
         lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat_3/in2");
         lgraph = connectLayers(lgraph,"concat_3","maxpool_Module3");
         lgraph = connectLayers(lgraph,"concat_3","concat3/in1");
         lgraph = connectLayers(lgraph,"relu_Module4_Level1","conv_Module4_Level2");
         lgraph = connectLayers(lgraph,"relu_Module4_Level1","concat_4/in1");
         lgraph = connectLayers(lgraph,"relu_Module4_Level2","concat_4/in2");
         lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in2");
         lgraph = connectLayers(lgraph,"relu_Module5_Level1","conv_Module5_Level2");
         lgraph = connectLayers(lgraph,"relu_Module5_Level1","concat_5/in1");
         lgraph = connectLayers(lgraph,"relu_Module5_Level2","concat_5/in2");
         lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in2");
         lgraph = connectLayers(lgraph,"relu_Module6_Level1","conv_Module6_Level2");
         lgraph = connectLayers(lgraph,"relu_Module6_Level1","concat_6/in1");
         lgraph = connectLayers(lgraph,"relu_Module6_Level2","concat_6/in2");
         lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in2");
         lgraph = connectLayers(lgraph,"relu_Module7_Level1","conv_Module7_Level2");
         lgraph = connectLayers(lgraph,"relu_Module7_Level1","concat/in2");
         lgraph = connectLayers(lgraph,"relu_Module7_Level2","concat/in1");

         plot(lgraph);
      end
      
      %%
      % load deepmedic DenseUnet, input: number of channels
%       function lgraph = LoadNNDeepMedic(NumberChannels)
%          disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 
%          lgraph = layerGraph();

%       end

%%
      % load 2D  , input: number of channels
      function lgraph = LoadNNDenseUnet2d(NumberChannels)
         
         disp('TODO') 
         % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR
         
         lgraph = layerGraph();
      
         tempLayers = [
             imageInputLayer([64 64 NumberChannels],"Name","input","Normalization","none")
             convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module1_Level1")
             reluLayer("Name","relu_Module1_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module1_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = depthConcatenationLayer(2,"Name","depthcat_1");
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module2_Level1")
             reluLayer("Name","relu_Module2_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module2_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = depthConcatenationLayer(2,"Name","depthcat_2");
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module3_Level1")
             reluLayer("Name","relu_Module3_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module3_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = depthConcatenationLayer(2,"Name","depthcat_3");
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_5")
             transposedConv2dLayer([2 2],512,"Name","transConv_Module4","Stride",[2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_4")
             convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_7")
             transposedConv2dLayer([2 2],256,"Name","transConv_Module5","Stride",[2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_6")
             convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_8")
             transposedConv2dLayer([2 2],128,"Name","transConv_Module6","Stride",[2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_9")
             convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
             reluLayer("Name","relu_Module7_Level1")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
             reluLayer("Name","relu_Module7_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             depthConcatenationLayer(2,"Name","depthcat_10")
             convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
             softmaxLayer("Name","softmax")
             dicePixelClassificationLayer("Name","output")];
         lgraph = addLayers(lgraph,tempLayers);

         % clean up helper variable
         clear tempLayers;
         % Connect all the branches of the network to create the network graph.

         lgraph = connectLayers(lgraph,"relu_Module1_Level1","conv_Module1_Level2");
         lgraph = connectLayers(lgraph,"relu_Module1_Level1","depthcat_1/in1");
         lgraph = connectLayers(lgraph,"relu_Module1_Level2","depthcat_1/in2");
         lgraph = connectLayers(lgraph,"depthcat_1","maxpool_Module1");
         lgraph = connectLayers(lgraph,"depthcat_1","depthcat_9/in1");
         lgraph = connectLayers(lgraph,"relu_Module2_Level1","conv_Module2_Level2");
         lgraph = connectLayers(lgraph,"relu_Module2_Level1","depthcat_2/in1");
         lgraph = connectLayers(lgraph,"relu_Module2_Level2","depthcat_2/in2");
         lgraph = connectLayers(lgraph,"depthcat_2","maxpool_Module2");
         lgraph = connectLayers(lgraph,"depthcat_2","depthcat_6/in2");
         lgraph = connectLayers(lgraph,"relu_Module3_Level1","conv_Module3_Level2");
         lgraph = connectLayers(lgraph,"relu_Module3_Level1","depthcat_3/in1");
         lgraph = connectLayers(lgraph,"relu_Module3_Level2","depthcat_3/in2");
         lgraph = connectLayers(lgraph,"depthcat_3","maxpool_Module3");
         lgraph = connectLayers(lgraph,"depthcat_3","depthcat_4/in1");
         lgraph = connectLayers(lgraph,"relu_Module4_Level1","conv_Module4_Level2");
         lgraph = connectLayers(lgraph,"relu_Module4_Level1","depthcat_5/in2");
         lgraph = connectLayers(lgraph,"relu_Module4_Level2","depthcat_5/in1");
         lgraph = connectLayers(lgraph,"transConv_Module4","depthcat_4/in2");
         lgraph = connectLayers(lgraph,"relu_Module5_Level1","conv_Module5_Level2");
         lgraph = connectLayers(lgraph,"relu_Module5_Level1","depthcat_7/in1");
         lgraph = connectLayers(lgraph,"relu_Module5_Level2","depthcat_7/in2");
         lgraph = connectLayers(lgraph,"transConv_Module5","depthcat_6/in1");
         lgraph = connectLayers(lgraph,"relu_Module6_Level1","conv_Module6_Level2");
         lgraph = connectLayers(lgraph,"relu_Module6_Level1","depthcat_8/in1");
         lgraph = connectLayers(lgraph,"relu_Module6_Level2","depthcat_8/in2");
         lgraph = connectLayers(lgraph,"transConv_Module6","depthcat_9/in2");
         lgraph = connectLayers(lgraph,"relu_Module7_Level1","conv_Module7_Level2");
         lgraph = connectLayers(lgraph,"relu_Module7_Level1","depthcat_10/in1");
         lgraph = connectLayers(lgraph,"relu_Module7_Level2","depthcat_10/in2");

         plot(lgraph);

      end
      
      %%
      % load 2D  , input: number of channels
      function lgraph = LoadNNUnet2d(NumberChannels)
         disp('TODO') % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR 

         lgraph = layerGraph();


         tempLayers = [
             imageInputLayer([64 64 NumberChannels],"Name","input","Normalization","none")
             convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module1_Level1")
             reluLayer("Name","relu_Module1_Level1")
             convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module1_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module2_Level1")
             reluLayer("Name","relu_Module2_Level1")
             convolution2dLayer([3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module2_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module3_Level1")
             reluLayer("Name","relu_Module3_Level1")
             convolution2dLayer([3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module3_Level2")];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level1")
             convolution2dLayer([3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level2")
             transposedConv2dLayer([2 2],512,"Name","transConv_Module4","Stride",[2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(3,2,"Name","concat3")
             convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level1")
             convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level2")
             transposedConv2dLayer([2 2],256,"Name","transConv_Module5","Stride",[2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(3,2,"Name","concat2")
             convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level1")
             convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level2")
             transposedConv2dLayer([2 2],128,"Name","transConv_Module6","Stride",[2 2])];
         lgraph = addLayers(lgraph,tempLayers);

         tempLayers = [
             concatenationLayer(3,2,"Name","concat1")
             convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
             reluLayer("Name","relu_Module7_Level1")
             convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
             reluLayer("Name","relu_Module7_Level2")
             convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
             softmaxLayer("Name","softmax")
             dicePixelClassificationLayer('Name','output')];
         lgraph = addLayers(lgraph,tempLayers);

         % clean up helper variable
         clear tempLayers;

         % Connect all the branches of the network to create the network graph.

         lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
         lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in2");
         lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
         lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in2");
         lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
         lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in2");
         lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in1");
         lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in1");
         lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in1");


         plot(lgraph);
      end
      % @amaleki101 @EGates1 @MBhatter @psarlashkar @RajiMR  - what other NN methods should we add ? 
   end
end

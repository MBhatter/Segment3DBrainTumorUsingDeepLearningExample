classdef ImageSegmentationUnet2D  < ImageSegmentationBaseClass  
   methods
      function loadneuralnet(obj,NumberChannels)
         tempLayers = [
             imageInputLayer([64 64 NumberChannels],"Name","input","Normalization","none")
             convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module1_Level1")
             reluLayer("Name","relu_Module1_Level1")
             convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module1_Level2")];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module2_Level1")
             reluLayer("Name","relu_Module2_Level1")
             convolution2dLayer([3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module2_Level2")];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
             batchNormalizationLayer("Name","BN_Module3_Level1")
             reluLayer("Name","relu_Module3_Level1")
             convolution2dLayer([3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module3_Level2")];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         tempLayers = [
             maxPooling2dLayer([2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2])
             convolution2dLayer([3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level1")
             convolution2dLayer([3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module4_Level2")
             transposedConv2dLayer([2 2],512,"Name","transConv_Module4","Stride",[2 2])];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         tempLayers = [
             concatenationLayer(3,2,"Name","concat3")
             convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level1")
             convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module5_Level2")
             transposedConv2dLayer([2 2],256,"Name","transConv_Module5","Stride",[2 2])];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         tempLayers = [
             concatenationLayer(3,2,"Name","concat2")
             convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level1")
             convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
             reluLayer("Name","relu_Module6_Level2")
             transposedConv2dLayer([2 2],128,"Name","transConv_Module6","Stride",[2 2])];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         tempLayers = [
             concatenationLayer(3,2,"Name","concat1")
             convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
             reluLayer("Name","relu_Module7_Level1")
             convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
             reluLayer("Name","relu_Module7_Level2")
             convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
             softmaxLayer("Name","softmax")
             dicePixelClassificationLayer('Name','output')];
         obj.lgraph = addLayers(obj.lgraph,tempLayers);
         
         % clean up helper variable
         clear tempLayers;
         % Connect Layer Branches
         % Connect all the branches of the network to create the network graph.
         
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module1_Level2","maxpool_Module1");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module1_Level2","concat1/in2");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module2_Level2","maxpool_Module2");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module2_Level2","concat2/in2");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module3_Level2","maxpool_Module3");
         obj.lgraph = connectLayers(obj.lgraph,"relu_Module3_Level2","concat3/in2");
         obj.lgraph = connectLayers(obj.lgraph,"transConv_Module4","concat3/in1");
         obj.lgraph = connectLayers(obj.lgraph,"transConv_Module5","concat2/in1");
         obj.lgraph = connectLayers(obj.lgraph,"transConv_Module6","concat1/in1");
         % Plot Layers
         
         plot(obj.lgraph);
      end
   end
end

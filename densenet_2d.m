%define n as number of channels
n = 1;

lgraph = layerGraph();

tempLayers = [
    imageInputLayer([64 64 n],"Name","input","Normalization","none")
    convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    reluLayer("Name","relu_Module1_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(3,2,"Name","concat_1");
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

tempLayers = concatenationLayer(3,2,"Name","concat_2");
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

tempLayers = concatenationLayer(3,2,"Name","concat_3");
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
    concatenationLayer(3,2,"Name","concat_4")
    transposedConv2dLayer([2 2],512,"Name","transConv_Module4","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat3")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_5")
    transposedConv2dLayer([2 2],256,"Name","transConv_Module5","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat2")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_6")
    transposedConv2dLayer([2 2],128,"Name","transConv_Module6","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat1")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat")
    convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassificationLayer("Name","output")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"relu_Module1_Level1","conv_Module1_Level2");
lgraph = connectLayers(lgraph,"relu_Module1_Level1","concat_1/in1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat_1/in2");
lgraph = connectLayers(lgraph,"concat_1","maxpool_Module1");
lgraph = connectLayers(lgraph,"concat_1","concat1/in2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","conv_Module2_Level2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","concat_2/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat_2/in2");
lgraph = connectLayers(lgraph,"concat_2","maxpool_Module2");
lgraph = connectLayers(lgraph,"concat_2","concat2/in2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","conv_Module3_Level2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","concat_3/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat_3/in2");
lgraph = connectLayers(lgraph,"concat_3","maxpool_Module3");
lgraph = connectLayers(lgraph,"concat_3","concat3/in2");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","conv_Module4_Level2");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","concat_4/in1");
lgraph = connectLayers(lgraph,"relu_Module4_Level2","concat_4/in2");
lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in1");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","conv_Module5_Level2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","concat_5/in1");
lgraph = connectLayers(lgraph,"relu_Module5_Level2","concat_5/in2");
lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in1");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","conv_Module6_Level2");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","concat_6/in1");
lgraph = connectLayers(lgraph,"relu_Module6_Level2","concat_6/in2");
lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in1");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","conv_Module7_Level2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","concat/in1");
lgraph = connectLayers(lgraph,"relu_Module7_Level2","concat/in2");

plot(lgraph);

%%

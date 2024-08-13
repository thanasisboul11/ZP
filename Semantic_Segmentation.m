path = uigetdir;

images = path + "\images";
masks = path + "\masks";

imdsTrain = imageDatastore(images,"ReadFcn",...
            @(x)imresize(rgb2gray(imread(x)),[256,256]));numClasses = 2;
classes = ["zp","bg"];
labels = [1 0];
pxdsTrain = pixelLabelDatastore(masks,classes,labels,"ReadFcn",@(x)im2bw(imresize(imread(x),[256 256])));

[ dsTrain, dsVal, dsTest ] = splitDataSet(imdsTrain,pxdsTrain);

tbl = countEachLabel(pxdsTrain);
numberPixels = sum(tbl.PixelCount);

frequency = tbl.PixelCount/numberPixels;
classWeights = 1 ./ frequency;


netlayers=[ 
    imageInputLayer([256 256 1]);
%--1d
    convolution2dLayer(3, 32, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2, 'Stride', 2);
%--2d
    convolution2dLayer(3, 128, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer('Name', 'relu2conv');
    maxPooling2dLayer(2, 'Stride', 2);
%--3d
    convolution2dLayer(3, 128, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer('Name', 'relu3conv');%<==== concatenate to 3u
    maxPooling2dLayer(2, 'Stride', 2);
%--4d
    convolution2dLayer(3, 512, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2, 'Stride', 2);
%------- Expanding path
%--4u
    transposedConv2dLayer([2 2], 512, 'Stride', 2);
    batchNormalizationLayer; 
    reluLayer();
    
%--3u
    transposedConv2dLayer([2 2], 128, 'Stride', 2);   
    concatenationLayer(3, 2, 'Name', 'concat3');%<========
%    convolution2dLayer(3, 128, 'Padding', 'same');
    batchNormalizationLayer;
    reluLayer();
%--2u
    transposedConv2dLayer([2 2], 128, 'Stride', 2);
    concatenationLayer(3, 2, 'Name', 'concat2');%<========
    
    batchNormalizationLayer;
    reluLayer();
%--1u
    transposedConv2dLayer([2 2], 32, 'Stride', 2);
    batchNormalizationLayer;
    reluLayer();

    convolution2dLayer(1, numClasses);

    softmaxLayer();
    pixelClassificationLayer('Classes', classes, 'ClassWeights', classWeights);
];


% Create layer graph
lgraph = layerGraph(netlayers);

% Connect relu3up to the second input of concat
lgraph = connectLayers(lgraph, 'relu3conv', 'concat3/in2');
% Connect relu3up to the second input of concat
lgraph = connectLayers(lgraph, 'relu2conv', 'concat2/in2');
% Analyze the network
analyzeNetwork(lgraph);

netlayers=lgraph;


options = trainingOptions('adam',...
    'InitialLearnRate',0.001,...
    'MaxEpochs',30,...
    'MiniBatchSize',25,...
    'Plots','training-progress',...
    'ValidationData',dsVal);



% netlayers = trainNetwork(dsTrain,netlayers,options);

% save('ZPnet.mat','netlayers');
load('ZPnet.mat','netlayers');


I = readimage(imdsTestSplit,3);
C = semanticseg(I,netlayers,Classes=classes);
B = labeloverlay(I,C);
imshow(B)



pxdsResults = semanticseg(dsTest,netlayers, ...
    Classes=classes, ...
    MiniBatchSize=4, ...
    WriteLocation=tempdir);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTestSplit,Verbose=false);

dataSetMetrics = metrics.DataSetMetrics
classMetrics = metrics.ClassMetrics
metrics.ImageMetrics
metrics.ConfusionMatrix
metrics.NormalizedConfusionMatrix
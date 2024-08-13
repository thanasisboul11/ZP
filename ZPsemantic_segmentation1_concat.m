path=uigetdir;
imageFolderTrain = path+"\images";
labelFolderTrain = path+"\masks";

imdsTrain = imageDatastore(imageFolderTrain,"ReadFcn",...
            @(x)imresize(rgb2gray(imread(x)),[256,256]));
numClasses=2;
classNames = ["zonepelu", "background"];
labels = [1 0];
pxdsTrain = pixelLabelDatastore(labelFolderTrain,classNames,labels,"ReadFcn",@(x)im2bw(imresize(imread(x),[256,256])));

tbl = countEachLabel(pxdsTrain);
numberPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / numberPixels;
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
    pixelClassificationLayer('Classes', classNames, 'ClassWeights', classWeights);
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

% Ορισμός παραμέτρων εκπαίδευσης
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',25,...
    'Plots','training-progress',...
    'Verbose',false);

%Εκπαίδευση δικτύου
trainingData = combine(imdsTrain,pxdsTrain);
netlayers = trainNetwork(trainingData,netlayers,options);

% Φορτώστε μια νέα εικόνα
newIm3ch=imread(path+"\images\sample2_frame_12.png");
newIm3ch=imresize(newIm3ch,[256,256]);
newImage=rgb2gray(newIm3ch);

% Πρόβλεψη ετικέτας κατηγορίας
[C,scores] = semanticseg(newImage,netlayers);
imzp=(C=="zonepelu"); %figure; imshow(imzp);

%======= Results Presentation ============================
zpedg = edge(imzp,'sobel');

% 
im = newIm3ch;figure; imshow(im)
[R, C, ~]=size(im);
 
for r=1:R
    for c=1:C
        if zpedg(r,c)==true
            im(r,c,1)=255;
            im(r,c,2)=0;
            im(r,c,3)=0;
        end
    end
end
figure;imshow(im)
%========= Saving of the net and weights saving =======
% Save the network structure and weights to a .mat file
save(path+"\matlabcode\netlayers_zonepelu_concat.mat",'netlayers');



function [dsTrain,dsVal,dsTest] = splitDataSet(imdsTrain,pxdsTrain,trainRatio,validationRatio,testRatio)

    if nargin < 4
        trainRatio = 0.7;
        validationRatio = 0.2;
        testRatio = 0.1;
    end
    
    
    numFiles = numel(imdsTrain.Files); % Total files in train datastore
    indices = randperm(numFiles); % Random indices for each file 
    
    
    numTrain = round(trainRatio * numFiles);  % = 144 files for training
    numVal = round(validationRatio * numFiles); % =41 files for validation
    numTest = numFiles - numTrain - numVal; % =20 files for testing our CNN
    
    % Split the indices into training, validation, and test sets
    trainIndices = indices(1:numTrain); % first 144 elements of the indices matrix for the 144 files, 1x144
    valIndices = indices(numTrain+1:numTrain+numVal); % 41 (144th-185th) elements of the indices matrix for the 41 files 
    testIndices = indices(numTrain+numVal+1:end); % 20 (186th - 205th) elements.....
    
    % Correspond each index to each png image
    imdsTrainSplit = subset(imdsTrain, trainIndices);
    imdsValSplit = subset(imdsTrain, valIndices);
    imdsTestSplit = subset(imdsTrain, testIndices);
    
    pxdsTrainSplit = subset(pxdsTrain, trainIndices);
    pxdsValSplit = subset(pxdsTrain, valIndices);
    pxdsTestSplit = subset(pxdsTrain, testIndices);
    
    % Combine images and labels for each dataset
    dsTrain = combine(imdsTrainSplit, pxdsTrainSplit);
    dsVal = combine(imdsValSplit, pxdsValSplit);
    dsTest = combine(imdsTestSplit, pxdsTestSplit);
end
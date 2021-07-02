function  outputLabel=FaceRecognition2(trainPath, testPath)
    % load training and test images
    imdsTrain = imageDatastore(trainPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
    imdsTest = imageDatastore(testPath);
    
    % save labels of training images
    labelImgSet = char(imdsTrain.Labels);
    outputLabel = [];
    warning('off')

    % load the pre-trained model downloaded from https://github.com/rcmalli/keras-vggface
    model_path = './vggface-model.h5'; 
    net = importKerasNetwork(model_path, 'OutputLayerType','classification');
    inputSize = net.Layers(1).InputSize;

    % resize training and test images to fit the model input size
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    
    % extract and normalize the values from the flatten layer of the pre-trained model
    trainFeatures = normalize(activations(net,augimdsTrain ,'flatten_1', 'OutputAs','rows'));
    testFeatures = normalize(activations(net,augimdsTest ,'flatten_1', 'OutputAs','rows'));
    
    % calculate cosine similarity of test image with all training images
    % assign label of the training image that corresponds to the largest cosine similarity value to test image
    for i = 1:size(testFeatures,1)
        scores = zeros(size(trainFeatures,1), 1);
        for j = 1:size(trainFeatures,1)
            scores(j, :) = cosine_sim(testFeatures(i,:), trainFeatures(j,:)');
        end
        max_idx = find(scores == max(scores));
        outputLabel = [outputLabel; labelImgSet(max_idx, :)];
    end
end

function Cs = cosine_sim(x, y)
xy   = dot(x,y);
nx   = norm(x);
ny   = norm(y);
nxny = nx*ny;
Cs   = xy/nxny;
end
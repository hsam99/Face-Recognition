function  outputLabel=FaceRecognition1(trainPath, testPath)
    % load training and test images
    train_image_path = [];
    folderNames=ls(trainPath);
    labelImgSet=folderNames(3:end,:);

    test_img_folder = ls(testPath);
    test_img_files = test_img_folder(3:end, :);
    outputLabel = [];

    for i = 1:length(labelImgSet)
        img_file = dir([trainPath, labelImgSet(i, :)]);
        train_image_path = [train_image_path; [trainPath, labelImgSet(i ,:), '\', img_file(3).name]];
    end

    train_features = zeros(length(train_image_path), 56576);
    test_features = zeros(length(test_img_files), 56576);

    % extract and normalize HOG features for each training image
    for i = 1:length(train_image_path)
        img = imread(train_image_path(i, :));
        gray_img = rgb2gray(img);
        gray_img = imresize(gray_img, [160 130]);
        hogTrain = extractHOGFeatures(gray_img, 'CellSize', [8 8], 'BlockSize', [4 4], 'BlockOverlap', [3 3], 'NumBins', 16);
        train_features(i, :) = normalize(hogTrain);
    end

    % extract and normalize HOG features for each test image
    for i = 1:length(test_img_files)
        img = imread([testPath, test_img_files(i, :)]);
        gray_img = rgb2gray(img);
        gray_img = imresize(gray_img, [160 130]);
        hogTest = extractHOGFeatures(gray_img, 'CellSize', [8 8], 'BlockSize', [4 4], 'BlockOverlap', [3 3], 'NumBins', 16);
        test_features(i, :) = normalize(hogTest);
    end

    % calculate cross correlation of test image with all training images
    % assign label of the training image that corresponds to the largest cross correlation value to test image
    for i = 1:size(test_features,1)
        scores = zeros(size(train_features,1), 1);
        for j = 1:size(train_features,1)
                scores(j, :) = test_features(i,:) * train_features(j,:)';
        end
        max_idx = find(scores == max(scores));
        outputLabel = [outputLabel; labelImgSet(max_idx, :)];
    end

end



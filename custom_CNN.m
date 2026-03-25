 
clc;
close all;
clear;

%% Load images
digitDatasetPath = fullfile('C:\Users\Tower Computers\Downloads\input images');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Determine the split up
total_split = countEachLabel(imds);

%% Visualize random images
num_images = length(imds.Labels);
perm = randperm(num_images, 6);
figure;
for idx = 1:length(perm)
    subplot(2, 3, idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s', imds.Labels(perm(idx))));
end

%% K-fold Validation
num_folds = 5;


for fold_idx = 1:num_folds
    fprintf('Processing %d among %d folds \n', fold_idx, num_folds);

    % Test Indices for current fold
    test_idx = fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds, test_idx);
    labeltest = countEachLabel(imdsTest);

    % Train indices for current fold
    train_idx = setdiff(1:num_images, test_idx);

    % Train cases for current fold
    imdsTrain = subset(imds, train_idx);
    labeltrain = countEachLabel(imdsTrain);

    % Custom CNN architecture
    layers = [
        imageInputLayer([227 227 3])
        convolution2dLayer(3, 16, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        convolution2dLayer(3, 32, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        convolution2dLayer(3, 64, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(8)
        softmaxLayer
        classificationLayer];

    % Training Options
    options = trainingOptions('rmsprop',...
        'ExecutionEnvironment', 'auto',...
        'MaxEpochs', 10, 'MiniBatchSize', 20,...
        'Shuffle', 'every-epoch', ...
        'InitialLearnRate', 1e-5, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Data Augmentation
    augmenter = imageDataAugmenter(...
        'RandRotation', [-5 5], 'RandXReflection', 1, 'RandXShear', [-0.05 0.05], 'RandYShear', [-0.05 0.05]);

    % Resizing all training images to [227 227]
    auimds = augmentedImageDatastore([227 227], imdsTrain, 'DataAugmentation', augmenter);

    % Training
    net = trainNetwork(auimds, layers, options);

    % Resizing all testing images to [227 227]
    augtestimds = augmentedImageDatastore([227 227], imdsTest);

    % Testing and their corresponding Labels and Posterior for each Case
    predicted_labels(test_idx) = classify(net, augtestimds);
end

%% Performance Study
% Actual Labels
actual_labels = imds.Labels;

% Confusion Matrix
confMat = confusionmat(actual_labels, predicted_labels);

% Display Confusion Matrix
figure;
confusionchart(confMat, unique(actual_labels), 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix:Customized CNN');
% True Positives, False Positives, False Negatives
tp = confMat(1, 1);
fp = confMat(2, 1);
fn = confMat(1, 2);

% Precision, Recall, F1-score
precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * ((precision * recall) / (precision + recall));
% Accuracy
accuracy = sum(predicted_labels == actual_labels) / numel(actual_labels);

% Displaying the results
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(precision)]);
disp(['Recall: ' num2str(recall)]);
disp(['F1-score: ' num2str(f1_score)]);

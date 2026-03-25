clc;close all;clear;%delete(findall(0));
%%cross validation
%load images
digitDatasetPath = fullfile('C:\Users\Tower Computers\Downloads\input images');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
% Visualize random images
perm=randperm(num_images,6);
figure;
for idx=1:length(perm)
    
    subplot(2,3,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end

%%K-fold Validation
% Number of folds
num_folds=5;

% Loop for each fold
for fold_idx=1:num_folds
    
    fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
    
   % Test Indices for current fold
    test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds,test_idx);
     labeltest=countEachLabel(imdsTest)
    % Train indices for current fold
    train_idx=setdiff(1:length(imds.Files),test_idx);
    
    % Train cases for current fold
    imdsTrain = subset(imds,train_idx);
 labeltrain= countEachLabel(imdsTrain)
    % alexNet Architecture 
    
     % Replacing the last layers with new layers
    layersTransfer = alexnet('Weights','none');

   
    numClasses = numel(categories(imdsTrain.Labels));
 layers = [
     layersTransfer(1:end-7)
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
     softmaxLayer
     classificationLayer];
    
    % Training Options, we choose a small mini-batch size due to limited images 
    options = trainingOptions('rmsprop',...
        'ExecutionEnvironment','auto',...
        'MaxEpochs',10,'MiniBatchSize',20,...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-5, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    % Data Augumentation
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    
   % Resizing all training images to [224 224] for ResNet architecture
    auimds = augmentedImageDatastore([227 227],imdsTrain,'DataAugmentation',augmenter);
    
    % Training
     netTransfer = trainNetwork(auimds,layers,options);
%    netTransfer = trainNetwork(imdsTrain,layers,options);
    % Resizing all testing images to [224 224] for ResNet architecture   
     augtestimds = augmentedImageDatastore([227 227],imdsTest);
   
    % Testing and their corresponding Labels and Posterior for each Case
     [predicted_labels(test_idx),posterior(test_idx,:)] = classify(netTransfer,augtestimds);
%    [predicted_labels(test_idx),posterior(test_idx,:)] = classify(netTransfer,imdsTest);
    
    
end
analyzeNetwork(netTransfer)
%%Performance Study
% Actual Labels
actual_labels=imds.Labels;
% Confusion Matrix
confMat = confusionmat(actual_labels, predicted_labels');

% Display Confusion Matrix
figure;
confusionchart(confMat, unique(actual_labels), 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix: AlexNet');
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


function [] = trainMain()
    [annotationMatrixPosTrain,annotationMatrixNegTrain,annotationMatrixPosTest,annotationMatrixNegTest] = PreProcessingData();

    categoryClassifier = bag_of_words();
    load('annoPosTrain.mat');
    load('annoNegTrain.mat');
    load('annoPosTest.mat');
    load('annoNegTest.mat');
    
    fNamePositive = 'train/positive/rp_';
    fNameNegative = 'train/negative/rn_';
    
    fTestPositive = 'test/positive/ep_';
    fTestNegative = 'test/negative/ep_';

    lbpVectorsTrainPositive = zeros(size(annotationMatrixPosTrain,1),9216,'double');
    lbpVectorsTrainNegative = zeros(size(annotationMatrixNegTrain,1),9216,'double');
    
    lbpVectorsTestPositive = zeros(size(annotationMatrixPosTest,1),9216,'double');
    lbpVectorsTestNegative = zeros(size(annotationMatrixNegTest,1),9216,'double');

    hogVectorsTrainPositive = zeros(size(annotationMatrixPosTrain,1),1764,'double');
    hogVectorsTrainNegative = zeros(size(annotationMatrixNegTrain,1),1764,'double');
    
    hogVectorsTestPositive = zeros(size(annotationMatrixPosTest,1),1764,'double');
    hogVectorsTestNegative = zeros(size(annotationMatrixNegTest,1),1764,'double');
    
    bovw_TrainPositive = zeros(size(annotationMatrixPosTrain,1),3,'double');
    bovw_TrainNegative = zeros(size(annotationMatrixNegTrain,1),3,'double');
    
    for i = 1:size(annotationMatrixPosTrain,1)
        fName = strcat(fNamePositive,num2str(i,'%05.f.jpg'));
        img = imread(fName);
        [labelIdx,scores] = predict(categoryClassifier,img);
        bovw_TrainPositive(i,1) = labelIdx;
        bovw_TrainPositive(i,2:3) = scores;
        LBP = extractlbpfeatures(img);
        HoG = extracthogfeatures(img);
        lbpVectorsTrainPositive(i,:) = LBP;
        hogVectorsTrainPositive(i,:) = HoG;
        fprintf('Positive BOVW LBP HOG %d\n',i);
    end

    for i = 1:size(annotationMatrixPosTest,1)
        fName = strcat(fTestPositive,num2str(i,'%05.f.jpg'));
        img = imread(fName);
        LBP = extractlbpfeatures(img);
        HoG = extracthogfeatures(img);
        lbpVectorsTestPositive(i,:) = LBP;
        hogVectorsTestPositive(i,:) = HoG;
        fprintf('Positive LBP HOG %d\n',i);
    end


    for i = 1:size(annotationMatrixNegTrain,1)
        fName = strcat(fNameNegative,num2str(i,'%05.f.jpg'));
        img = imread(fName);
        [labelIdx,scores] = predict(categoryClassifier,img);
        bovw_TrainNegative(i,1) = labelIdx;
        bovw_TrainNegative(i,2:3) = scores;
        LBP = extractlbpfeatures(img);
        HoG = extracthogfeatures(img);
        lbpVectorsTrainNegative(i,:) = LBP;
        hogVectorsTrainNegative(i,:) = HoG;
        fprintf('Negative BOVW LBP HOG %d\n',i);
    end

    for i = 1:size(annotationMatrixNegTest,1)
        fName = strcat(fTestNegative,num2str(i,'%05.f.jpg'));
        img = imread(fName);
        LBP = extractlbpfeatures(img);
        HoG = extracthogfeatures(img);
        lbpVectorsTestNegative(i,:) = LBP;
        hogVectorsTestNegative(i,:) = HoG;
        fprintf('Negative LBP HOG %d\n',i);
    end

    save('lbpTrainPos.mat','lbpVectorsTrainPositive');
    save('lbpTrainNeg.mat','lbpVectorsTrainNegative');
    save('hogTrainPos.mat','hogVectorsTrainPositive');
    save('hogTrainNeg.mat','hogVectorsTrainNegative');
    save('bovwTrainPos.mat','bovw_TrainPositive');
    save('bovwTrainNeg.mat','bovw_TrainNegative');
    
    save('lbpTestPos.mat','lbpVectorsTestPositive');
    save('lbpTestNeg.mat','lbpVectorsTestNegative');
    save('hogTestPos.mat','hogVectorsTestPositive');
    save('hogTestNeg.mat','hogVectorsTestNegative');

    annoPosTrain = size(annotationMatrixPosTrain,1);
    annoNegTrain = size(annotationMatrixNegTrain,1);
    datasetLandMarkTrain();    %Creating Image Dataset Matrix for Annotation Training
    output = trainClassificationMod(annoPosTrain,annoNegTrain);
    %Annotation Training (Already trained and present with the code, very slow to train)
    %annoOutput = TrainingLandMark();
end
function [outputFinal] = trainClassificationMod( annoPosTrain, annoNegTrain )
    %Calculating Target Class
    true_class = uint8(zeros(annoPosTrain+300,2));
    true_class(1:annoPosTrain,1) = 1;
    true_class(annoPosTrain+1:annoPosTrain+300,2) = 1;

    %Generating random Indeces
    itemIndex = randperm(annoNegTrain,300);

    %Loading Annotation Points
    load('overall_annotation');
    annotationTrainPositive = input_annotation(1:annoPosTrain,:);

    %Loading HoG Features
    load('hogTrainPos.mat');
    load('hogTrainNeg.mat');

    %Loading Local Binary Patterns
    load('lbpTrainPos.mat');
    load('lbpTrainNeg.mat');
    
    %Loading Local BoVW Labels
    load('bovwTrainPos.mat');
    load('bovwTrainNeg.mat');

    annotationTrainNegative_subset = zeros(300,74);
    hogVectorsTrainNegative_subset = zeros(300,1764);
    lbpVectorsTrainNegative_subset = zeros(300,9216); %59 for MATLAB 9216 for Implemented
    bovw_TrainNegative_subset = zeros(300,3);

    %Selecting Negative Subsets
    for i = 1:300
        annotationTrainNegative_subset(i,:) = input_annotation(itemIndex(i)+annoPosTrain,:);
        hogVectorsTrainNegative_subset(i,:) = hogVectorsTrainNegative(itemIndex(i),:);
        lbpVectorsTrainNegative_subset(i,:) = lbpVectorsTrainNegative(itemIndex(i),:);
        bovw_TrainNegative_subset(i,:) = bovw_TrainNegative(itemIndex(i),:);
    end

    %Meaging Positive Class and Negative Class
    annotationVectors = [annotationTrainPositive;annotationTrainNegative_subset];
    hogVectors = [hogVectorsTrainPositive;hogVectorsTrainNegative_subset];
    lbpVectors = [lbpVectorsTrainPositive;lbpVectorsTrainNegative_subset];
    bovwVectors = [bovw_TrainPositive;bovw_TrainNegative_subset];
    
    %Claculating Distances Among Annotation Points
    itr = 37*73;
    annotationDistances = zeros(annoPosTrain+300,itr);
    annotationX = zeros(annoPosTrain+300,37);
    annotationY = zeros(annoPosTrain+300,37);
    for k = 1:annoPosTrain+300
        j = 1;
        for i = 1:2:74
            annotationX(k,j) = annotationVectors(k,i);
            annotationY(k,j) = annotationVectors(k,i+1);
            j = j + 1;
        end
    end
    for k = 1:annoPosTrain+300
        pos = 1;
        for i = 1:37
            for j = 1:i-1
                distVector = double([annotationX(k,j),annotationY(k,j);annotationX(k,i),annotationY(k,i)]);
                annotationDistances(k,pos) = pdist(distVector);
                pos = pos + 1;
            end
        end
    end
    %disp(size(annotationDistances));
    %disp(size(bovwVectors));
    %disp(size(hogVectors));
    %disp(size(lbpVectors));
    inputVector = [annotationDistances';bovwVectors';hogVectors';lbpVectors'];
    
    disp('Training Started')
    %Training Final Neural Network
    net = patternnet([100,10]);
    net.trainParam.epochs = 100;
    net.divideParam.trainRatio = 85/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 5/100;
    net.trainParam.max_fail = 15;
    trainedNet = train(net,inputVector,true_class','useParallel','yes');
    outputFinal = trainedNet(inputVector);
    save trainedNet;

    output_class = vec2ind(outputFinal);
    output_class = output_class';

    count = 0;
    for i = 1:581
        if (i<=281)
            if output_class(i)==1
                count = count + 1;
            end
        end
        if (i>281)
            if output_class(i)==2
                count = count + 1;
            end
        end
    end
    accuracy = count/(annoPosTrain+300);
    disp('Training Accuracy');
    disp(accuracy);
end
function [output] = TrainingLandMark()
    load('overall_annotation');
    load('imageList_Train');
    sampleImages = zeros(1000,4096);
    sampleTargets = zeros(1000,74);
    sampleIndex = randperm(12271,1000);
    sampleIndex = sort(sampleIndex);

    j = 1;
    for i = 1:12271
        if ismember(i,sampleIndex) == 1
          sampleImages(j,:) = imgList(i,:);
          sampleTargets(j,:) = input_annotation(i,:);
          j = j+1;
        end
    end

    disp('Training Started');
    net = feedforwardnet([600,100,20]);
    net.trainParam.epochs = 300;
    trainedNetAnnotation = train(net,sampleImages',sampleTargets','useParallel','yes','useGPU','yes');
    output = trainedNetAnnotation(sampleImages');
    save 'trainedNetAnnotation';
end
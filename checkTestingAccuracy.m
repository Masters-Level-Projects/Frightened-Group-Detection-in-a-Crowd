function [ ] = checkTestingAccuracy( trainedNetAnnotation,trainedNet,categoryClassifier )
    Neg = dir('TestGroup/Negative');
    Pos = dir('TestGroup/Positive');
    %images = uint8(zeros(size(Pos,1)+size(Neg,1)-4,))
    count = 0;
    match_count = 0;
    for i = 3:size(Pos,1)
        count = count + 1;
        imName = strcat('TestGroup/Positive/',Pos(i).name);
        img =  imread(imName);
        fprintf('Fetch Image %d\n',count);
        disp(Pos(i).name)
        [classVal,img_new] = groupEmotionAccuracyCheck(img,trainedNetAnnotation,trainedNet,categoryClassifier);
        if classVal ~= 0
            if classVal == 1
                match_count = match_count + 1;
            end
            if classVal == -1
                classVal = 2;
            end
            save_path = strcat('TestGroup/output/',int2str(classVal));
            save_path = strcat(save_path,'_');
            save_path = strcat(save_path,int2str(count));
            save_path = strcat(save_path,'.jpg');
            imwrite(img_new,save_path)
        end
    end
    for i = 3:size(Neg,1)
        count = count + 1;
        imName = strcat('TestGroup/Negative/',Neg(i).name);
        img =  imread(imName);
        fprintf('Fetch Image %d\n',count);
        disp(Neg(i).name)
        [classVal,img_new] = groupEmotionAccuracyCheck(img,trainedNetAnnotation,trainedNet,categoryClassifier);
        if classVal ~= 0
            if classVal == -1
                match_count = match_count + 1;
            end
            if classVal == -1
                classVal = 2;
            end
            save_path = strcat('TestGroup/output/',int2str(classVal));
            save_path = strcat(save_path,'_');
            save_path = strcat(save_path,int2str(count));
            save_path = strcat(save_path,'.jpg');
            imwrite(img_new,save_path)
        end
    end
    accuracy = (match_count/count)*100;
    disp('Accuracy is');
    disp(accuracy);
end
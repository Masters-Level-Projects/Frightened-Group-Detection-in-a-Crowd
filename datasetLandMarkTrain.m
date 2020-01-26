function [] = datasetLandMarkTrain()
    load('annoPosTrain.mat');
    load('annoNegTrain.mat');

    fNamePositive = 'train/positive/rp_';
    fNameNegative = 'train/negative/rn_';
    imgList = zeros(12271,4096);
    k = 1;
    for i = 1:size(annotationMatrixPosTrain,1)
        fName = strcat(fNamePositive,num2str(i,'%05.f.jpg'));
        img = imread(fName);
        disp(k);
        [len,brd] = size(img);
        p = 1;
        imRow = zeros(1,4096);
        for j = 1:len
            for l = 1:brd
                imRow(1,p) = img(j,l);
            end
        end
        imgList(k,:) = imRow;
        k = k + 1;
    end
    for i = 1:size(annotationMatrixNegTrain,1)
        fName = strcat(fNameNegative,num2str(i,'%05.f.jpg'));
        img = imread(fName);
        disp(k);
        [len,brd] = size(img);
        p = 1;
        imRow = zeros(1,4096);
        for j = 1:len
            for l = 1:brd
                imRow(1,p) = img(j,l);
                p = p + 1;
            end
        end
        imgList(k,:) = imRow;
        k = k + 1;
    end
    save('imageList_Train','imgList');
    
    C = [annotationMatrixPosTrain;annotationMatrixNegTrain];
    D = zeros(size(C,1),74);
    for i = 1:size(C,1)
        d_i = 1;
        for j = 1:37
            D(i,d_i) = C(i,j,1);
            d_i = d_i + 1;
            D(i,d_i) = C(i,j,2);
            d_i = d_i + 1;
        end
    end
    input_annotation = D;
    save('overall_annotation','input_annotation');
end
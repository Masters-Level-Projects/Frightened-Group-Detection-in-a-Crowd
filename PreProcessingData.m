function [annotationMatrixPosTrain,annotationMatrixNegTrain,annotationMatrixPosTest,annotationMatrixNegTest] = PreProcessingData()
    fileID = fopen('EmoLabel/list_partition_label.txt','r');
    A = fscanf(fileID,'%s');
    B = strrep(A,'.jpg','.jpg,');
    B = strrep(B,'train',' train');
    B = strrep(B,'test',' test');
    C = strsplit(B,' ');
    listFiles = '';
    [len,brd] = size(C);
    classnumber = zeros(1,brd-1);
    for i = 2:brd
        temp = C{i};
        strTemp = strsplit(temp,',');
        listFiles = strcat(listFiles,strTemp{1});
        listFiles = strcat(listFiles,',');
        classnumber(1,i-1) = str2double(strTemp{2});
        fprintf('Fetch Image %d\n',i);
    end
    countPosTrain = 0;
    countNegTrain = 0;
    countPosTest = 0;
    countNegTest = 0;
    for i = 1:brd-1
        if (classnumber(1,i)==2)&&(C{i+1}(2)== 'r')
            countPosTrain = countPosTrain + 1;
        elseif (classnumber(1,i)~=2)&&(C{i+1}(2)== 'r')
            countNegTrain = countNegTrain + 1;
        elseif (classnumber(1,i)==2)&&(C{i+1}(2)== 'e')
            countPosTest = countPosTest + 1;
        elseif (classnumber(1,i)~=2)&&(C{i+1}(2)== 'e')
            countNegTest = countNegTest + 1;
        end
    end
    D = strrep(listFiles,'.jpg','_auto_attri.txt ');
    E = strsplit(D,' ,');
    annotationMatrixPosTrain = zeros(countPosTrain,37,2);
    annotationMatrixPosTest = zeros(countPosTest,37,2);
    annotationMatrixNegTrain = zeros(countNegTrain,37,2);
    annotationMatrixNegTest = zeros(countNegTest,37,2);
    [lenE,brdE] = size(E);
    k1 = 1;
    k2 = 1;
    k3 = 1;
    k4 = 1;
    for i = 1:brdE-1
        temp = E{i};
        temp = strcat('Annotation/auto/',temp);
        ffileID = fopen(temp,'r');
        item = fscanf(ffileID,'%f');
        interestSize = size(item());
        if ((classnumber(1,i)==2)&&(E{i}(2)=='r'))
            for j = 1:interestSize(1)
                calrem = rem(j,2);
                annotationMatrixPosTrain(k1,uint8(j/2),2-calrem)=uint16(item(j));
            end
            k1 = k1 + 1;
        elseif ((classnumber(1,i)~=2)&&(E{i}(2)=='r'))
            for j = 1:interestSize(1)
                calrem = rem(j,2);
                annotationMatrixNegTrain(k2,uint8(j/2),2-calrem)=uint16(item(j));
            end
            k2 = k2 + 1;
        elseif ((classnumber(1,i)==2)&&(E{i}(2)=='e'))
            for j = 1:interestSize(1)
                calrem = rem(j,2);
                annotationMatrixPosTest(k3,uint8(j/2),2-calrem)=uint16(item(j));
            end
            k3 = k3 + 1;
        elseif ((classnumber(1,i)~=2)&&(E{i}(2)=='e'))
            for j = 1:interestSize(1)
                calrem = rem(j,2);
                annotationMatrixNegTest(k4,uint8(j/2),2-calrem)=uint16(item(j));
            end
            k4 = k4 + 1;
        end
        fclose(ffileID);
        fprintf('Annotation %d\n',i);
    end
    boundingBoxMatrixPosTrain = zeros(countPosTrain,4);
    boundingBoxMatrixPosTest = zeros(countPosTest,4);
    boundingBoxMatrixNegTrain = zeros(countNegTrain,4);
    boundingBoxMatrixNegTest = zeros(countNegTest,4);
    F = strrep(listFiles,'.jpg','_boundingbox.txt ');
    G = strsplit(F,' ,');
    [lenG,brdG] = size(G);
    k1 = 1;
    k2 = 1;
    k3 = 1;
    k4 = 1;
    for i = 1:brdG-1
        temp = G{i};
        temp = strcat('Annotation/boundingbox/',temp);
        ffileID = fopen(temp,'r');
        item = fscanf(ffileID,'%f');
        if (classnumber(1,i)==2)&&(G{i}(2)== 'r')
            for j = 1:4
                boundingBoxMatrixPosTrain(k1,j)=item(j);
            end
            k1 = k1 + 1;
        elseif (classnumber(1,i)~=2)&&(G{i}(2)== 'r')
            for j = 1:4
                boundingBoxMatrixNegTrain(k2,j)=item(j);
            end
            k2 = k2 + 1;
        elseif (classnumber(1,i)==2)&&(G{i}(2)== 'e')
            for j = 1:4
                boundingBoxMatrixPosTest(k3,j)=item(j);
            end
            k3 = k3 + 1;
        elseif (classnumber(1,i)~=2)&&(G{i}(2)== 'e')
            for j = 1:4
                boundingBoxMatrixNegTest(k4,j)=item(j);
            end
            k4 = k4 + 1;
        end
        fclose(ffileID);
        fprintf('Bounding Box %d\n',i);
    end
    boundingBoxMatrixPosTrain = uint16(boundingBoxMatrixPosTrain);
    boundingBoxMatrixPosTest = uint16(boundingBoxMatrixPosTest);
    boundingBoxMatrixNegTrain = uint16(boundingBoxMatrixNegTrain);
    boundingBoxMatrixNegTest = uint16(boundingBoxMatrixNegTest);
    H = strsplit(listFiles,',');
    [lenH,brdH] = size(H);

    k1 = 1;
    k2 = 1;
    k3 = 1;
    k4 = 1;
    for k = 1:brdH-1
        flag = 0;
        if (classnumber(1,k)==2)&&(H{k}(2)== 'r')
            flag = 1;
        elseif (classnumber(1,k)~=2)&&(H{k}(2)== 'r')
            flag = 2;
        elseif (classnumber(1,k)==2)&&(H{k}(2)== 'e')
            flag = 3;
        elseif (classnumber(1,k)~=2)&&(H{k}(2)== 'e')
            flag = 4;
        end
        mypath = H{k};
        mypath = strcat('Image/original/',mypath);
        img = imread(mypath);
        if size(img,3)==3
            img = rgb2gray(img);
        end
        if flag == 1
            for i = 1:37
                annotationMatrixPosTrain(k1,i,1) = annotationMatrixPosTrain(k1,i,1)-boundingBoxMatrixPosTrain(k1,1);
                annotationMatrixPosTrain(k1,i,2) = annotationMatrixPosTrain(k1,i,2)-boundingBoxMatrixPosTrain(k1,2);
            end
        end
        if flag == 2
            for i = 1:37
                annotationMatrixNegTrain(k2,i,1) = annotationMatrixNegTrain(k2,i,1)-boundingBoxMatrixNegTrain(k2,1);
                annotationMatrixNegTrain(k2,i,2) = annotationMatrixNegTrain(k2,i,2)-boundingBoxMatrixNegTrain(k2,2);
            end
        end
        if flag == 3
            for i = 1:37
                annotationMatrixPosTest(k3,i,1) = annotationMatrixPosTest(k3,i,1)-boundingBoxMatrixPosTest(k3,1);
                annotationMatrixPosTest(k3,i,2) = annotationMatrixPosTest(k3,i,2)-boundingBoxMatrixPosTest(k3,2);
            end
        end
        if flag == 4
            for i = 1:37
                annotationMatrixNegTest(k4,i,1) = annotationMatrixNegTest(k4,i,1)-boundingBoxMatrixNegTest(k4,1);
                annotationMatrixNegTest(k4,i,2) = annotationMatrixNegTest(k4,i,2)-boundingBoxMatrixNegTest(k4,2);
            end
        end
        if flag == 1
            img = imcrop(img,[boundingBoxMatrixPosTrain(k1,1),boundingBoxMatrixPosTrain(k1,2),boundingBoxMatrixPosTrain(k1,3)-boundingBoxMatrixPosTrain(k1,1),boundingBoxMatrixPosTrain(k1,4)-boundingBoxMatrixPosTrain(k1,2)]);
        end
        if flag == 2
            img = imcrop(img,[boundingBoxMatrixNegTrain(k2,1),boundingBoxMatrixNegTrain(k2,2),boundingBoxMatrixNegTrain(k2,3)-boundingBoxMatrixNegTrain(k2,1),boundingBoxMatrixNegTrain(k2,4)-boundingBoxMatrixNegTrain(k2,2)]);
        end
        if flag == 3
            img = imcrop(img,[boundingBoxMatrixPosTest(k3,1),boundingBoxMatrixPosTest(k3,2),boundingBoxMatrixPosTest(k3,3)-boundingBoxMatrixPosTest(k3,1),boundingBoxMatrixPosTest(k3,4)-boundingBoxMatrixPosTest(k3,2)]);
        end
        if flag == 4
            img = imcrop(img,[boundingBoxMatrixNegTest(k4,1),boundingBoxMatrixNegTest(k4,2),boundingBoxMatrixNegTest(k4,3)-boundingBoxMatrixNegTest(k4,1),boundingBoxMatrixNegTest(k4,4)-boundingBoxMatrixNegTest(k4,2)]);
        end
        [imlen,imbrd] = size(img);
        imgSize = 64;
        img = imresize(img,[imgSize,imgSize]);
        if flag == 1
            for i = 1:37
                annotationMatrixPosTrain(k1,i,1) = uint16(annotationMatrixPosTrain(k1,i,1)*imgSize/imbrd);
                annotationMatrixPosTrain(k1,i,2) = uint16(annotationMatrixPosTrain(k1,i,2)*imgSize/imlen);
            end
        end
        if flag == 2
            for i = 1:37
                annotationMatrixNegTrain(k2,i,1) = uint16(annotationMatrixNegTrain(k2,i,1)*imgSize/imbrd);
                annotationMatrixNegTrain(k2,i,2) = uint16(annotationMatrixNegTrain(k2,i,2)*imgSize/imlen);
            end
        end
        if flag == 3
            for i = 1:37
                annotationMatrixPosTest(k3,i,1) = uint16(annotationMatrixPosTest(k3,i,1)*imgSize/imbrd);
                annotationMatrixPosTest(k3,i,2) = uint16(annotationMatrixPosTest(k3,i,2)*imgSize/imlen);
            end
        end
        if flag == 4
            for i = 1:37
                annotationMatrixNegTest(k4,i,1) = uint16(annotationMatrixNegTest(k4,i,1)*imgSize/imbrd);
                annotationMatrixNegTest(k4,i,2) = uint16(annotationMatrixNegTest(k4,i,2)*imgSize/imlen);
            end
        end
        if flag == 1
            for i = 1:37
                y = uint16(annotationMatrixPosTrain(k1,i,1));
                x = uint16(annotationMatrixPosTrain(k1,i,2));
                if x<1
                    x=1;
                end
                if y<1
                    y=1;
                end
                if x>imgSize
                    x=imgSize;
                end
                if y>imgSize
                    y=imgSize;
                end
                annotationMatrixPosTrain(k1,i,1) = y;
                annotationMatrixPosTrain(k1,i,2) = x;
            end
        end
        if flag == 2
            for i = 1:37
                y = uint16(annotationMatrixNegTrain(k2,i,1));
                x = uint16(annotationMatrixNegTrain(k2,i,2));
                if x<1
                    x=1;
                end
                if y<1
                    y=1;
                end
                if x>imgSize
                    x=imgSize;
                end
                if y>imgSize
                    y=imgSize;
                end
                annotationMatrixNegTrain(k2,i,1) = y;
                annotationMatrixNegTrain(k2,i,2) = x;
            end
        end
        if flag == 3
            for i = 1:37
                y = uint16(annotationMatrixPosTest(k3,i,1));
                x = uint16(annotationMatrixPosTest(k3,i,2));
                if x<1
                    x=1;
                end
                if y<1
                    y=1;
                end
                if x>imgSize
                    x=imgSize;
                end
                if y>imgSize
                    y=imgSize;
                end
                annotationMatrixPosTest(k3,i,1) = y;
                annotationMatrixPosTest(k3,i,2) = x;
            end
        end
        if flag == 4
            for i = 1:37
                y = uint16(annotationMatrixNegTest(k4,i,1));
                x = uint16(annotationMatrixNegTest(k4,i,2));
                if x<1
                    x=1;
                end
                if y<1
                    y=1;
                end
                if x>imgSize
                    x=imgSize;
                end
                if y>imgSize
                    y=imgSize;
                end
                annotationMatrixNegTest(k4,i,1) = y;
                annotationMatrixNegTest(k4,i,2) = x;
            end
        end
        fprintf('Operations %d\n',k);
        if flag == 1
            fName = strcat('rp_',num2str(k1,'%05.f.jpg'));
            fName = strcat('train/positive/',fName);
            k1 = k1 + 1;
        end
        if flag == 2
            fName = strcat('rn_',num2str(k2,'%05.f.jpg'));
            fName = strcat('train/negative/',fName);
            k2 = k2 + 1;
        end
        if flag == 3
            fName = strcat('ep_',num2str(k3,'%05.f.jpg'));
            fName = strcat('test/positive/',fName);
            k3 = k3 + 1;
        end
        if flag == 4
            fName = strcat('ep_',num2str(k4,'%05.f.jpg'));
            fName = strcat('test/negative/',fName);
            k4 = k4 + 1;
        end
        imwrite(img,fName);
    end
    save('annoPosTrain.mat','annotationMatrixPosTrain');
    save('annoNegTrain.mat','annotationMatrixNegTrain');
    save('annoPosTest.mat','annotationMatrixPosTest');
    save('annoNegTest.mat','annotationMatrixNegTest');
end
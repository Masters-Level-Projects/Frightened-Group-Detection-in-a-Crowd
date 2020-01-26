function [ classValue,img2 ] = groupEmotionAccuracyCheck( img,trainedNetAnnotation,trainedNet,categoryClassifier )
    %[len,brd] = uigetfile('*.*');
    %img = imread([brd,len]);
    k = 1;
    faceDetector = vision.CascadeObjectDetector();
    BB = step(faceDetector,img);
    if size(BB,1)==0
        %disp('No Face Detected');
        classValue = 0;
        img2 = 0;
    elseif size(BB,1)==1
        %disp('Only One Face Detected');
        centrePoints = zeros(1,2);
        centrePoints(1,1) = uint32(BB(1,1)+BB(1,1)+BB(1,3));
        centrePoints(1,2) = uint32(BB(1,2)+BB(1,2)+BB(1,4));
        [classValue,faceFearIntensity] = calculateGroupEmotion(img,1,centrePoints(1,:),1,trainedNetAnnotation,trainedNet,categoryClassifier);
        img2 = img;
        if classValue == 1
            bbTitle = 'Feared Face';
        else
            bbTitle = 'Un-feared Face';
        end
        img2(BB(1,2):BB(1,2)+2,BB(1,1):BB(1,1)+BB(1,3),2:3)=0;
        img2(BB(1,2):BB(1,2)+2,BB(1,1):BB(1,1)+BB(1,3),1)=255;
        img2(BB(1,2)+BB(1,4)-2:BB(1,2)+BB(1,4),BB(1,1):BB(1,1)+BB(1,3),2:3)=0;
        img2(BB(1,2)+BB(1,4)-2:BB(1,2)+BB(1,4),BB(1,1):BB(1,1)+BB(1,3),1)=255;
        img2(BB(1,2):BB(1,2)+BB(1,4),BB(1,1):BB(1,1)+2,2:3)=0;
        img2(BB(1,2):BB(1,2)+BB(1,4),BB(1,1):BB(1,1)+2,1)=255;
        img2(BB(1,2):BB(1,2)+BB(1,4),BB(1,1)+BB(1,3)-2:BB(1,1)+BB(1,3),2:3)=0;
        img2(BB(1,2):BB(1,2)+BB(1,4),BB(1,1)+BB(1,3)-2:BB(1,1)+BB(1,3),1)=255;
        img2 = insertText(img2,[BB(1,1)-10, BB(1,2)-10],bbTitle,'FontSize',20);
        %disp(classValue);
        %disp(faceFearIntensity);
    else
        %disp('Multiple Faces Detected');
        %disp(size(BB,1));
        faceCentersY = zeros(size(BB,1),1);
        faceCentersX = zeros(size(BB,1),1);
        meanRadii = zeros(size(BB,1),1);
        for i = 1:size(BB,1)
            faceCentersY(i,1) = uint32((BB(i,1)+ BB(i,1)+BB(i,3))/2);
            faceCentersX(i,2) = uint32((BB(i,2)+ BB(i,2)+BB(i,4))/2);
            meanRadiiVector = double([faceCentersY(i,1),faceCentersX(i,1);BB(i,1),BB(i,2)]);
            meanRadii(i,1) = (pdist(meanRadiiVector));
        end
        count = size(BB,1);
        closenessValueVector = zeros(uint16(count*(count-1)/2),1);
        paramCount = 1;

        %Computing Preliminary Similarities
        for i = 1:count
            for j = 1:i-1
                %parameterVector = double([faceCentersY(i,1),faceCentersX(i,1),meanRadii(i,1);
                                          %faceCentersY(j,1),faceCentersX(j,1),meanRadii(j,1)]);
                parameterVector = double([faceCentersY(i,1),faceCentersX(i,1);
                                          faceCentersY(j,1),faceCentersX(j,1)]);
                closenessValueVector(paramCount,1) = pdist(parameterVector);%/(meanRadii(i,1)+meanRadii(j,1));
                paramCount = paramCount + 1;
            end
        end

        %Selecting k-cluster Maximal Splitting EigenVectors
        if k <= 1
            %disp('All individuals are a single cluster');
            idx = uint8(ones(count,1));
            centrePoints = uint32(zeros(k,2));
            classMinY = uint32(zeros(k,2));
            classMinY(:,1) = size(img,2);
            classMinX = uint32(zeros(k,1));
            classMinX(:,1) = size(img,1);
            classMaxY = uint32(zeros(k,1));
            classMaxY(:,1) = 1;
            classMaxX = uint32(zeros(k,1));
            classMaxX(:,1) = 1;
            for i = 1:size(BB,1)
                temp = idx(i,1);
                if classMinY(temp,1)>BB(i,1)
                    classMinY(temp,1) = BB(i,1);
                end
                if classMinX(temp,1)>BB(i,2)
                    classMinX(temp,1) = BB(i,2);
                end
                if classMaxY(temp,1)<(BB(i,1)+BB(i,3))
                    classMaxY(temp,1) = BB(i,1)+BB(i,3);
                end
                if classMaxX(temp,1)<(BB(i,2)+BB(i,4))
                    classMaxX(temp,1) = BB(i,2)+BB(i,4);
                end
            end
            centrePoints(1,1) = uint32((classMinY(1,1)+classMaxY(1,1))/2);
            centrePoints(1,2) =  uint32((classMinX(1,1)+classMaxX(1,1))/2);
            cornerPoints = uint32(zeros(1,4));
            cornerPoints(1,1) = classMinX(1,1);
            cornerPoints(1,2) = classMinY(1,1);
            cornerPoints(1,3) = classMaxX(1,1);
            cornerPoints(1,4) = classMaxY(1,1);

        end

        %[classValue,faceFearIntensity] = calculateGroupEmotion(img,idx,centrePoints(i,:),i);
        img2 = img;
        [classValue,faceFearIntensity] = calculateGroupEmotion(img,idx,centrePoints(1,:),1,trainedNetAnnotation,trainedNet,categoryClassifier);
        if classValue() == 1
            bbTitle = 'Feared Group';
        else
            bbTitle = 'Un-Feared Group';
        end
        %rectangle('Position',[cornerPoints(i,2)-5, cornerPoints(i,1)-5, cornerPoints(i,4)-cornerPoints(i,2)+5, cornerPoints(i,3)-cornerPoints(i,1)+5],'Curvature',0.2,'EdgeColor','y','LineWidth', 2)
        img2(cornerPoints(1,1):cornerPoints(1,1)+4,cornerPoints(1,2):cornerPoints(1,4),1:2)=0;
        img2(cornerPoints(1,1):cornerPoints(1,1)+4,cornerPoints(1,2):cornerPoints(1,4),3)=255;
        img2(cornerPoints(1,3)-4:cornerPoints(1,3),cornerPoints(1,2):cornerPoints(1,4),1:2)=0;
        img2(cornerPoints(1,3)-4:cornerPoints(1,3),cornerPoints(1,2):cornerPoints(1,4),3)=255;
        img2(cornerPoints(1,1):cornerPoints(1,3),cornerPoints(1,2):cornerPoints(1,2)+4,1:2)=0;
        img2(cornerPoints(1,1):cornerPoints(1,3),cornerPoints(1,2):cornerPoints(1,2)+4,3)=255;
        img2(cornerPoints(1,1):cornerPoints(1,3),cornerPoints(1,4)-4:cornerPoints(1,4),1:2)=0;
        img2(cornerPoints(1,1):cornerPoints(1,3),cornerPoints(1,4)-4:cornerPoints(1,4),3)=255;
        img2 = insertText(img2,[cornerPoints(1,2)-25, cornerPoints(1,1)-25],bbTitle,'FontSize',20);
        for j = 1:count
            img2(BB(j,2):BB(j,2)+2,BB(j,1):BB(j,1)+BB(j,3),2:3)=0;
            img2(BB(j,2):BB(j,2)+2,BB(j,1):BB(j,1)+BB(j,3),1)=255;
            img2(BB(j,2)+BB(j,4)-2:BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+BB(j,3),2:3)=0;
            img2(BB(j,2)+BB(j,4)-2:BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+BB(j,3),1)=255;
            img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+2,2:3)=0;
            img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+2,1)=255;
            img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1)+BB(j,3)-2:BB(j,1)+BB(j,3),2:3)=0;
            img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1)+BB(j,3)-2:BB(j,1)+BB(j,3),1)=255;
            img2 = insertText(img2,[BB(j,1)+10, BB(j,2)+10],faceFearIntensity(j),'FontSize',16,'BoxColor','green');
        end  
    end
end
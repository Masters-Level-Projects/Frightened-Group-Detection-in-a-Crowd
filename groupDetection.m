function [  ] = groupDetection( trainedNetAnnotation,trainedNet,categoryClassifier )
    %img = imread('pos113.jpg');
    [len,brd] = uigetfile('*.*');
    img = imread([brd,len]);
    %k = 3;
    faceDetector = vision.CascadeObjectDetector();
    BB = step(faceDetector,img);
    if size(BB,1)==0
        disp('No Face Detected');
    elseif size(BB,1)==1
        disp('Only One Face Detected');
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
        figure;
        imshow(img2);
        disp(classValue);
        disp(faceFearIntensity);
    else
        disp('Multiple Faces Detected');
        disp(size(BB,1));
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

        %Computing Standard Deviation of Similarity
        stdSimilarity = std(closenessValueVector);
        std_sigma2 = 2*stdSimilarity*stdSimilarity; % stdSimilarity or 2*stdSimilarity could also work

        %Calculating Adjacency_Matrix
        adjacency_matrix = zeros(count);
        for i = 1:count
            for j = 1:i-1
                %parameterVector = double([faceCentersY(i,1),faceCentersX(i,1),meanRadii(i,1);
                                          %faceCentersY(j,1),faceCentersX(j,1),meanRadii(j,1)]);
                parameterVector = double([faceCentersY(i,1),faceCentersX(i,1);
                                          faceCentersY(j,1),faceCentersX(j,1)]);
                closenessValue = pdist(parameterVector);%/(meanRadii(i,1)+meanRadii(j,1));
                adjacency_matrix(i,j) = exp(-closenessValue/std_sigma2);
                adjacency_matrix(j,i) = exp(-closenessValue/std_sigma2);
            end
        end

        %Calculating Degree Matrix
        degreeMatrix = zeros(count);
        for i = 1:count
            degreeMatrix(i,i) = sum(adjacency_matrix(i,:));
        end

        %Calculating Laplacian Matrix
        LaplacianMatrix = degreeMatrix-adjacency_matrix;

        %Normalizing Laplacian Matrix
        D_inv = inv(degreeMatrix).^(1/2);
        if count == 2
            LaplacianNormalized = LaplacianMatrix;
        else
            LaplacianNormalized = D_inv * LaplacianMatrix * D_inv;
        end
        %disp(LaplacianNormalized)
        %Finding EigenValues and EigenVectors
        %disp(Laplacian);
        [eigVectors,eigValues] = eig(LaplacianNormalized);
        vectEigValues = diag(eigValues);
        derivativeEigValues = diff(vectEigValues');
        eigenVectorsCluster = eigVectors(:,2:size(eigVectors,2));
        itr_max = min(count,3);
        eva = evalclusters(eigenVectorsCluster,'kmeans','CalinskiHarabasz','KList',[1:itr_max]);
        k = eva.OptimalK;
        fprintf('K %d\n',k);
        %Selecting k-cluster Maximal Splitting EigenVectors
        if k <= 1
            disp('All individuals are a single cluster');
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
            for i = 1:k
                centrePoints(i,1) = uint32((classMinY(i,1)+classMaxY(i,1))/2);
                centrePoints(i,2) =  uint32((classMinX(i,1)+classMaxX(i,1))/2);
            end
            cornerPoints = uint32(zeros(k,4));
            for i = 1:k
                cornerPoints(i,1) = classMinX(i,1);
                cornerPoints(i,2) = classMinY(i,1);
                cornerPoints(i,3) = classMaxX(i,1);
                cornerPoints(i,4) = classMaxY(i,1);
            end

        elseif k == count
            disp('All individuals are unrelated');
            idx = uint8(zeros(count,1));
            for i = 1:count
                idx(i) = i;
            end
            centrePoints = uint32(zeros(k,2));
            for i = 1:k
                centrePoints(i,1) = uint32((BB(i,1)+ BB(i,1)+BB(i,3))/2);
                centrePoints(i,2) = uint32((BB(i,2)+ BB(i,2)+BB(i,4))/2);
            end  
            cornerPoints = uint32(zeros(k,4));
            for i = 1:k
                cornerPoints(i,1) = BB(i,2);
                cornerPoints(i,2) = BB(i,1);
                cornerPoints(i,3) = BB(i,2)+BB(i,4);
                cornerPoints(i,4) = BB(i,1)+BB(i,3);
            end

        elseif k == 2
            disp('Cluster of 2 can be calculated using the second smallest eigenvector.');
            maxEigVectors = eigVectors(:,2);
            idx = uint8(zeros(count,1));
            for i = 1:count
                if maxEigVectors(i,1)>0
                    idx(i,1) = 1;
                else
                    idx(i,1) = 2;
                end
            end
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
            for i = 1:k
                centrePoints(i,1) = uint32((classMinY(i,1)+classMaxY(i,1))/2);
                centrePoints(i,2) =  uint32((classMinX(i,1)+classMaxX(i,1))/2);
            end
            cornerPoints = uint32(zeros(k,4));
            for i = 1:k
                cornerPoints(i,1) = classMinX(i,1);
                cornerPoints(i,2) = classMinY(i,1);
                cornerPoints(i,3) = classMaxX(i,1);
                cornerPoints(i,4) = classMaxY(i,1);
            end

        else
            disp('Calculating for Multiple Cluster');
            maxEigVectors = eigVectors(:,(2:k+1));
            %Normalizing Maximal Splitting EigenVectors
            for i = 1:count
                maxItem = max(maxEigVectors(i,:));
                maxEigVectors(i,:) = maxEigVectors(i,:)/maxItem;
            end
            disp('Running K-Means');
            [idx,~,kDistances] = kmeans(maxEigVectors,k);
            sumKD = sum(kDistances);
            %itr = uint16(count/k)+k;
            %Running K-Means Algorithm for Multiple Times
            for i = 1:7
                [idxTemp,~,kDistances] = kmeans(maxEigVectors,k);
                if (sumKD>sum(kDistances))
                    sumKD = sum(kDistances);
                    idx = idxTemp;
                end
            end
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
            for i = 1:k
                centrePoints(i,1) = uint32((classMinY(i,1)+classMaxY(i,1))/2);
                centrePoints(i,2) =  uint32((classMinX(i,1)+classMaxX(i,1))/2);
            end
            cornerPoints = uint32(zeros(k,4));
            for i = 1:k
                cornerPoints(i,1) = classMinX(i,1);
                cornerPoints(i,2) = classMinY(i,1);
                cornerPoints(i,3) = classMaxX(i,1);
                cornerPoints(i,4) = classMaxY(i,1);
            end
        end
        %[classValue,faceFearIntensity] = calculateGroupEmotion(img,idx,centrePoints(i,:),i);
        img2 = img;
        for i = 1:k
            %disp(img)
            %disp(idx)
            %disp(centrePoints(i,:))
            %disp(i)
            %disp(trainedNetAnnotation)
            %disp(trainedNet)
            %disp(categoryClassifier)
            [classValue,faceFearIntensity] = calculateGroupEmotion(img,idx,centrePoints(i,:),i,trainedNetAnnotation,trainedNet,categoryClassifier);
            if classValue() == 1
                bbTitle = 'Feared Group';
            else
                bbTitle = 'Un-Feared Group';
            end
            %rectangle('Position',[cornerPoints(i,2)-5, cornerPoints(i,1)-5, cornerPoints(i,4)-cornerPoints(i,2)+5, cornerPoints(i,3)-cornerPoints(i,1)+5],'Curvature',0.2,'EdgeColor','y','LineWidth', 2)
            img2(cornerPoints(i,1):cornerPoints(i,1)+4,cornerPoints(i,2):cornerPoints(i,4),1:2)=0;
            img2(cornerPoints(i,1):cornerPoints(i,1)+4,cornerPoints(i,2):cornerPoints(i,4),3)=255;
            img2(cornerPoints(i,3)-4:cornerPoints(i,3),cornerPoints(i,2):cornerPoints(i,4),1:2)=0;
            img2(cornerPoints(i,3)-4:cornerPoints(i,3),cornerPoints(i,2):cornerPoints(i,4),3)=255;
            img2(cornerPoints(i,1):cornerPoints(i,3),cornerPoints(i,2):cornerPoints(i,2)+4,1:2)=0;
            img2(cornerPoints(i,1):cornerPoints(i,3),cornerPoints(i,2):cornerPoints(i,2)+4,3)=255;
            img2(cornerPoints(i,1):cornerPoints(i,3),cornerPoints(i,4)-4:cornerPoints(i,4),1:2)=0;
            img2(cornerPoints(i,1):cornerPoints(i,3),cornerPoints(i,4)-4:cornerPoints(i,4),3)=255;
            img2 = insertText(img2,[cornerPoints(i,2)-25, cornerPoints(i,1)-25],bbTitle,'FontSize',20);
            for j = 1:count
                %rectangle('Position',[BB(j,1)-3, BB(j,2)-3, BB(j,3)+3, BB(j,4)+3],'Curvature',0.7,'EdgeColor','r','LineWidth', 1)
                if idx(j)==i
                    img2(BB(j,2):BB(j,2)+2,BB(j,1):BB(j,1)+BB(j,3),2:3)=0;
                    img2(BB(j,2):BB(j,2)+2,BB(j,1):BB(j,1)+BB(j,3),1)=255;
                    img2(BB(j,2)+BB(j,4)-2:BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+BB(j,3),2:3)=0;
                    img2(BB(j,2)+BB(j,4)-2:BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+BB(j,3),1)=255;
                    img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+2,2:3)=0;
                    img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1):BB(j,1)+2,1)=255;
                    img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1)+BB(j,3)-2:BB(j,1)+BB(j,3),2:3)=0;
                    img2(BB(j,2):BB(j,2)+BB(j,4),BB(j,1)+BB(j,3)-2:BB(j,1)+BB(j,3),1)=255;
                    img2 = insertText(img2,[BB(j,1), BB(j,2)],faceFearIntensity(j),'FontSize',8,'BoxColor','green');
                end
            end
            disp('Class');
            disp(bbTitle);
        end
    figure;
    imshow(img2)    
    end
end
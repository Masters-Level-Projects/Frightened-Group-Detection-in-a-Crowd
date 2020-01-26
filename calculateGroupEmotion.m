function [ group_class,faceFearIntensity ] = calculateGroupEmotion( img,idx,groupCentre,clusterNumber,trainedNetAnnotation,trainedNet,categoryClassifier )
    
    %Detecting Faces
    faceDetector = vision.CascadeObjectDetector();
    BB = step(faceDetector,img);

    %Counting Number of Faces
    count = size(BB,1);
    %Paramenters needs to be calculated
    areaFaces = zeros(1,count);
    centerDistanceFaces = zeros(1,count);
    faceClasses = zeros(1,count);
    softMaxVals = zeros(1,count);   %It should give the confidance with which faces are classified
    faceFearIntensity = zeros(1,count);

    %Group Parameters
    groupArea = (size(img,1)*size(img,2));
    %maxDistance = (((size(img,1)/2).^2)+((size(img,2)/2).^2)).^0.5;
    maxDistanceVector = double([0,0;uint8(size(img,1)/2),uint32(size(img,2)/2)]);
    maxDistance = pdist(maxDistanceVector);


    %Calculating Face Parameters
    for i = 1:size(BB,1)
        if idx(i) == clusterNumber
            areaFaces(1,i) = BB(i,3)*BB(i*4);
            faceCentreY = uint32((BB(i,1)+ BB(i,1)+BB(i,3))/2);
            faceCentreX = uint32((BB(i,2)+ BB(i,2)+BB(i,4))/2);
            distanceVector = double([groupCentre(1,1),groupCentre(1,2); faceCentreY,faceCentreX]);
            %[groupCentre(1,1),faceCentreY],[groupCentre(1,2),faceCentreX]
            centerDistanceFaces(1,i) = pdist(distanceVector);
            J = imcrop(img,BB(i,:));
            J = imresize(J,[64,64]);
            if size(J,3)>1
                J = rgb2gray(J);
            end
            %disp(i);
            [classVal,confidance] = findClass(J,trainedNetAnnotation,trainedNet,categoryClassifier);
            %faceClasses(1,i) = randsample([1,-1],1);    %Note here we'd calculate the class of the face
            %softMaxVals(1,i) = (0.5 + 0.5.*(rand(1)));  %Note here we'd take the softMaxVal of the class chosen
            faceClasses(1,i) = classVal;
            softMaxVals(1,i) = max(confidance);
            faceFearIntensity(1,i) = (areaFaces(1,i)/groupArea)*(1-(centerDistanceFaces(1,i)/maxDistance))*(faceClasses(1,i)*softMaxVals(1,i));
        end
    end

    %Summing-up classification Intensities
    sumFearedIntensity = sum(faceFearIntensity);
    
    %Classification
    if sumFearedIntensity>0
        group_class = 1;
    else
        group_class = -1;
    end
    %{
    figure,imshow(img)
    hold on
    for i = 1:count
        if idx(i) == clusterNumber
            areaFaces(1,i) = BB(i,3)*BB(i*4);
            faceCentreY = uint32((BB(i,1)+ BB(i,1)+BB(i,3))/2);
            faceCentreX = uint32((BB(i,2)+ BB(i,2)+BB(i,4))/2);
            line([groupCentre(1,1),faceCentreY],[groupCentre(1,2),faceCentreX])
        end
    end
    %}
end
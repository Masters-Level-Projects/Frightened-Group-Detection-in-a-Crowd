function [ output_class,outputValue ] = findClass( img,trainedNetAnnotation,trainedNet,categoryClassifier )

    imgRow = reshape(img,[1,4096]);
    
    annotation = trainedNetAnnotation(imgRow');
    annotation = annotation';
    LBP = extractlbpfeatures(img);
    HoG = extracthogfeatures(img);
    lbpVectors(1,:) = LBP;
    hogVectors(1,:) = HoG;
    
    
    [labelIdx,scores] = predict(categoryClassifier,img);
    bovw_Vector = zeros(1,3,'double');
    bovw_Vector(1,1) = labelIdx;
    bovw_Vector(1,2:3) = scores;

    %disp('Testing1');
    %Calculating Distances Among Annotation Points
    itr = 37*73;
    annotationDistances = zeros(1,itr);
    annotationX = zeros(1,37);
    annotationY = zeros(1,37);
    j = 1;
    for i = 1:2:74
        %disp(size(annotationX));
        %disp(size(annotation));
        annotationX(1,j) = annotation(1,i);
        annotationY(1,j) = annotation(1,i+1);
        j = j + 1;
    end
    pos = 1;
    for i = 1:37
        for j = 1:i-1
            distVector = double([annotationX(1,j),annotationY(1,j);annotationX(1,i),annotationY(1,i)]);
            annotationDistances(1,pos) = pdist(distVector);
            pos = pos + 1;
        end
    end
    inputVector = [annotationDistances';bovw_Vector';hogVectors';lbpVectors'];
    %disp('Testing2');
    %Calculating Class
    outputValue = trainedNet(inputVector);
    output_class = vec2ind(outputValue);
    if output_class == 2
        output_class = -1;
    end
    
end
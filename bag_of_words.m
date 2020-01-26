function [categoryClassifier] = bag_of_words()
    pathFolder = 'bag_of_words/train';
    imageSets = [imageSet(fullfile(pathFolder,'positive')), imageSet(fullfile(pathFolder,'negative'))]
    bag = bagOfFeatures(imageSets)
    categoryClassifier = trainImageCategoryClassifier(imageSets,bag)
    %img = imread('rn_00176.jpg');
    %img = rgb2gray(img);
    %[labelIdx,scores] = predict(categoryClassifier,img)
    save categoryClassifier;
end


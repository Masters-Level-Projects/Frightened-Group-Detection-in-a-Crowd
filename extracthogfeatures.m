function HoG = extracthogfeatures(origImg)
    
    cellSize = 8;   % 4X4 px cells in image
    blockSize = 16;  % 8X8 blocks with 4 cells
    binSize = 40;   % for intervals in histogram such that there are 9 bins
    total_Ang = 360;
    
    %grayImg = rgb2gray(origImg); %0 to 255 values
    doubImg = im2double(origImg); % 0 to 1
    workImg = padarray(doubImg, [1 1], 0, 'both');      %padding border with zeroes to calculate gradient at border
    
    function Ix = dabax(img)
        mask = [-1 0 1; ...
                -1 0 1; ...
                -1 0 1];
        Ix = conv2(img, mask, 'same');
    end

    function Iy = dabay(img)
        mask = [-1 -1 -1; ...
                 0 0 0; ...
                 1 1 1];
        Iy = conv2(img, mask, 'same');
    end

    function nval = l2norm(vect)
        sum = 0;
        for indx = 1:numel(vect)
            sum = sum + (vect(indx)^2);
        end
        nval = sqrt(sum);
    end

    %----------------------------------------------------------------------
    % STEP 1: Intensity Gradients, gradient magnitude, and gradient angles
    
    img_x_grad = dabax(workImg);    % compute the gradient in x direction
    img_x_grad = img_x_grad(2:size(img_x_grad, 1), 2:size(img_x_grad, 2));    % remove border pixels
    img_y_grad = dabay(workImg);    % compute the gradient in y direction
    img_y_grad = img_y_grad(2:size(img_y_grad, 1), 2:size(img_y_grad, 2));    % remove border pixels
    
    magnitude = ((img_x_grad .^ 2) + (img_y_grad .^ 2)) .^ 0.5;     % compute the magnitude matrix
    theta = (180.0/pi) * atan2(img_y_grad, img_x_grad);      % signed angle between the y-axis gradient and the x-axis gradient
    
    theta = bsxfun(@plus, theta, +180); % above between -180 to 180, making from 0 to 360
    
    %----------------------------------------------------------------------
    % STEP 2: 
    
    % applying gaussian spatial weighing to magnitude, in lieu of
    % interpolation during binning
    h = fspecial('gaussian', [cellSize cellSize], cellSize);
    magnitude=imfilter(magnitude,h);
    
    [h, w]=size(theta); %in our case that would be 100X100
    
    % following variables tell me how many 4X4 cells I will have in my image
    cellNumI=floor(h/cellSize);     % in vertical direction
    cellNumJ=floor(w/cellSize);     % in horizontal direction

    cell_histo=zeros(cellNumI,cellNumJ,total_Ang/binSize); % a 3d matirx of dim 25X25X9
    
    for cellI=1:cellNumI  % over all 25 cells given 100 X 100 px image
        for cellJ=1:cellNumJ

            for bin=1:total_Ang/binSize % for each cell selected above, over all 9 bins
                %  selecting cell pixels
                startI=(cellI-1)*cellSize+1;    % in vertical direction
                endI=(cellI)*cellSize;
                startJ=(cellJ-1)*cellSize+1;    % in horizontal direction
                endJ=(cellJ)*cellSize;

                % A=zeros(endI-startI+1,endJ-startJ+1);   % a temp matrix of size 4X4 px - the cell
                for i=startI:endI   %for each of 4 pixel vertically
                    for j=startJ:endJ %for each of 4 pixel horizontally
            % bins: (0-39), (40-79), (80-119), (120-159), (160-199), (200-239), (240-279), (280-319), (320-359)
                        if (theta(i, j) == total_Ang)       % handling the boundary case when theta = 360 degree
                            cell_histo(cellI,cellJ,1) = cell_histo(cellI,cellJ,1) + (magnitude(i, j)/2);
                            cell_histo(cellI,cellJ,total_Ang/binSize) = cell_histo(cellI,cellJ,total_Ang/binSize) + (magnitude(i, j)/2);
                        elseif ((theta(i,j)>=(bin-1)*binSize)&&(theta(i,j)<(bin)*binSize))            % otherwise simply summing up for histogram
                            cell_histo(cellI,cellJ,bin) = cell_histo(cellI,cellJ,bin) + magnitude(i, j);
                        end
                    end
                end
            end
        end
    end

    %----------------------------------------------------------------------
    % STEP 3: 8 X 8 Block normalization
    
    block_histo = double.empty(0,0);
    
    for blockIStart=1:cellNumI-1  % over all 8X8 blocks
        for blockJStart=1:cellNumJ-1
            blockVect = zeros(1, 4*(total_Ang/binSize));
            vect_i = 1;
            for cellI=0:blockSize/cellSize - 1 % for each of 4 cells in block selected above
                for cellJ=0:blockSize/cellSize - 1
                    blockVect(vect_i:vect_i+(total_Ang/binSize)-1) = cell_histo(blockIStart+cellI, blockJStart+cellJ, :);
                    vect_i = vect_i + (total_Ang/binSize);
                end
            end
            blockVect_norm = l2norm(blockVect);
            normalised_blockVect = blockVect ./ blockVect_norm;
            block_histo = [block_histo normalised_blockVect];
        end
    end
    
    HoG = block_histo;
end
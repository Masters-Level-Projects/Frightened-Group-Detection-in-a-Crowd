function LBP = extractlbpfeatures(img)
    % origImg is a grayscale image
    
    function nval = l2norm(vect)
        sum = 0;
        for indx = 1:numel(vect)
            sum = sum + (vect(indx)^2);
        end
        nval = sqrt(sum);
    end
    
    [len,brd] = size(img);
    paddedImg = uint8(zeros(len+2,brd+2));
    paddedImg(2:len+1,2:brd+1) = img;
    lbpImg = uint8(zeros(len,brd));
    for i = 2:len+1
        for j = 2:brd+1
            binaryList = uint8(zeros(8,1));
            if paddedImg(i,j)<=paddedImg(i-1,j-1)
                binaryList(1,1) = 1;
            else
                binaryList(1,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i-1,j)
                binaryList(2,1) = 1;
            else
                binaryList(2,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i-1,j+1)
                binaryList(3,1) = 1;
            else
                binaryList(3,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i,j+1)
                binaryList(4,1) = 1;
            else
                binaryList(4,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i+1,j+1)
                binaryList(5,1) = 1;
            else
                binaryList(5,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i+1,j)
                binaryList(6,1) = 1;
            else
                binaryList(6,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i+1,j-1)
                binaryList(7,1) = 1;
            else
                binaryList(7,1) = 0;
            end
            if paddedImg(i,j)<=paddedImg(i,j-1)
                binaryList(8,1) = 1;
            else
                binaryList(8,1) = 0;
            end
            sumLBP = 0;
            for k = 1:8
                sumLBP = sumLBP + binaryList(k,1)*(2^(8-k));
            end
            lbpImg(i-1,j-1) = sumLBP;
        end
    end
    
    %----------------------------------------------------------------------
    % Step 2: Histogram and block normalization
    
    cellSize = 16;   % 4X4 px cells in image
    blockSize = 32;  % 8X8 blocks with 4 cells
    numbkts = 256;   % for intervals in histogram such that there are 256 bins
    
    [h, w]=size(lbpImg); %in our case that would be 100X100
    
    % following variables tell me how many 4X4 cells I will have in my image
    cellNumI=floor(h/cellSize);     % in vertical direction
    cellNumJ=floor(w/cellSize);     % in horizontal direction

    cell_histo=zeros(cellNumI,cellNumJ,numbkts); % a 3d matirx of dim 25X25X9
    
    for cellI=1:cellNumI  % over all 16 cells given 128 X 128 px image
        for cellJ=1:cellNumJ

            for bin=1:numbkts % for each cell selected above, over all 256 bins
                %  selecting cell pixels
                startI=(cellI-1)*cellSize+1;    % in vertical direction
                endI=(cellI)*cellSize;
                startJ=(cellJ-1)*cellSize+1;    % in horizontal direction
                endJ=(cellJ)*cellSize;

                for i=startI:endI   %for each of 4 pixel vertically
                    for j=startJ:endJ %for each of 4 pixel horizontally
                        if ( lbpImg(i,j) == bin )            % simply summing up for histogram
                            cell_histo(cellI,cellJ,bin) = cell_histo(cellI,cellJ,bin) + 1;
                        end
                    end
                end
            end
        end
    end
    %size(cell_histo)
    %----------------------------------------------------------------------
    % STEP 3: 16 X 16 Block normalization
    
    block_histo = double.empty(0,0);
    
    for blockIStart=1:cellNumI-1  % over all 16X16 blocks
        for blockJStart=1:cellNumJ-1
            blockVect = zeros(1, 4*numbkts);
            vect_i = 1;
            for cellI=0:blockSize/cellSize - 1 % for each of 4 cells in block selected above
                for cellJ=0:blockSize/cellSize - 1
                    blockVect(vect_i:vect_i+numbkts-1) = cell_histo(blockIStart+cellI, blockJStart+cellJ, :);
                    vect_i = vect_i + numbkts;
                end
            end
            blockVect_norm = l2norm(blockVect);
            normalised_blockVect = blockVect ./ blockVect_norm;
            block_histo = [block_histo normalised_blockVect];
        end
    end
    
    LBP = block_histo;
end
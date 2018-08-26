% Load images from Imageset folder
imgSetVector = imageSet('Imageset','recursive');
montage(imgSetVector.ImageLocation);

% Detect BRISK features from first image
I = imread(imgSetVector.ImageLocation{1,1});
grayImage = rgb2gray(I);
points = detectBRISKFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

numImages = size(imgSetVector.ImageLocation,2);
tforms(numImages) = projective2d(eye(3));

for n = 2:numImages

    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;

    % Read I(n).
    I = imread(imgSetVector.ImageLocation{1,n});

    % Detect and extract BRISK features for I(n).
    grayImage = rgb2gray(I);
    points = detectBRISKFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);

    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

    % Compute T(1) * ... * T(n-1) * T(n)
    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

imageSize = size(I);

% Compute the output limits for each transform
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

% Find image appearing in centre
avgXLim = mean(xlim, 2);
[~, idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

% Apply inverse transform to all transforms
Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = Tinv.T * tforms(i).T;
end

% Recalculate output limits
for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
end

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([imageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([imageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Create a 2-D image-space as per the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

prevI = imread(imgSetVector.ImageLocation{1,1});
% Transform I into the panorama.
prevWarpedImage = imwarp(prevI, tforms(1), 'OutputView', panoramaView);
% Generate a binary mask.
prevMask = imwarp(true(size(prevI,1),size(prevI,2)), tforms(1), 'OutputView', panoramaView);

% Median filter neighbourhood size
m = 3;n = 3;

for i = 2:numImages

	% Generate transformed image and corresponding mask
    Ii = imread(imgSetVector.ImageLocation{1,i});
    warpedImagei = imwarp(Ii, tforms(i), 'OutputView', panoramaView);
    maski = imwarp(true(size(Ii,1),size(Ii,2)), tforms(i), 'OutputView', panoramaView);

	% Find overlapping area of two images ((i-1)th and ith image)
    commonMask = double(maski & prevMask);
    newmask1 = zeros(size(commonMask,1),size(commonMask,2),3);
    newmask1(:,:,1) = double(prevMask(:,:));
    newmask1(:,:,2) = double(prevMask(:,:));
    newmask1(:,:,3) = double(prevMask(:,:));
    newmask2 = zeros(size(commonMask,1),size(commonMask,2),3);
    newmask2(:,:,1) = double(maski(:,:));
    newmask2(:,:,2) = double(maski(:,:));
    newmask2(:,:,3) = double(maski(:,:));
    
	% Creating masks with linear intensity distribution of overlapping regions
    for k = 1:size(commonMask,1)
        j = 1;
        while j <= size(commonMask,2) & commonMask(k,j) == 0
            j=j+1;
        end
        left = j;
        while j <= size(commonMask,2) & commonMask(k,j) == 1
            j=j+1;
        end
        right = j-1;
        if left < right
            diff = 1/(right-left+1);
            value = double(1);
            for j = left:right
                newmask1(k,j,:) = value;
                newmask2(k,j,:) = 1-value;
                value = value - diff;
            end
        end
    end
    
	% Creating median filtered images from transformed images
    medfiltImage1r = medfilt2(prevWarpedImage(:,:,1),[m n]);
    medfiltImage1g = medfilt2(prevWarpedImage(:,:,2),[m n]);
    medfiltImage1b = medfilt2(prevWarpedImage(:,:,3),[m n]);
    medfiltImage2r = medfilt2(warpedImagei(:,:,1),[m n]);
    medfiltImage2g = medfilt2(warpedImagei(:,:,2),[m n]);
    medfiltImage2b = medfilt2(warpedImagei(:,:,3),[m n]);
    medfiltImage1 = cat(3,medfiltImage1r,medfiltImage1g,medfiltImage1b);
    medfiltImage2 = cat(3,medfiltImage2r,medfiltImage2g,medfiltImage2b);
    
	% Blending of images with varying intensity and median filtering
    X = (uint8(( (double(prevWarpedImage).*(newmask1>=0.5).*newmask1) + (double(warpedImagei).*(newmask2>=0.5).*newmask2) + (double(medfiltImage1).*(newmask1<0.5&newmask1>0).*newmask1) + (double(medfiltImage2).*(newmask2<0.5&newmask2>0).*newmask2))));
    
	% Initializing variables for next iteration
    prevMask = maski | prevMask;
    prevWarpedImage = X;
    
end

% Display mosaiced image
figure;
imshow(X);
imwrite(X,'mosaic.jpg');
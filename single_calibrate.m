
  %% load dataset
%images = imageDatastore(fullfile(toolboxdir('vision'),'visiondata', ...
%      'calibration', 'slr'));
%READSIZE = 5;
%images = imageDatastore('/home/cranesoar/Documents/calibrate dataset/Calibration Dataset','ReadSize', READSIZE);
images = imageDatastore('./Calibration Dataset/single camera');
%% 
[imagePoints,boardSize] = detectCheckerboardPoints(images.Files);
  %% set square size 'mm'
squareSize = 60;
worldPoints = generateCheckerboardPoints(boardSize, squareSize);
%% 
I = readimage(images,1); 
imageSize = [size(I,1), size(I,2)];

%cameraParams = estimateCameraParameters(imagePoints,worldPoints, ...
%                              'ImageSize',imageSize);
%% 
% compute the initial "guess" of intrinisc and extrinsic camera parameters
% Solve for the camera intriniscs and extrinsics in closed form ignoring
% distortion.

% Compute homographies for all images
%[H, validIdx] = computeHomographies(imagePoints, worldPoints);
w1 = warning('Error', 'MATLAB:nearlySingularMatrix'); %#ok
w2 = warning('Error', 'images:maketform:conditionNumberofAIsHigh'); %#ok

numImages = size(imagePoints, 3);
validIdx = true(numImages, 1);
homographies = zeros(3, 3, numImages);
for i = 1:numImages
    try    
        homographies(:, :, i) = ...
            computeHomography(double(imagePoints(:, :, i)), worldPoints);
    catch 
        validIdx(i) = false;
    end
end
warning(w1);
warning(w2);
homographies = homographies(:, :, validIdx);
if ~all(validIdx)
    warning(message('vision:calibrate:invalidHomographies', ...
        numImages - size(homographies, 3), numImages));
end

if size(homographies, 3) < 2
    error(message('vision:calibrate:notEnoughValidHomographies'));
end

imagePoints = imagePoints(:, :, validIdx);  %% validIdx = imageused
%% solve initIntrinsics  //compute B accroding the num of n
% 
% if isempty(initIntrinsics)
    if ~isempty(imageSize)  %%????? n==1   ?????
        % assume zero skew and centered principal point for initial guess
        cx = (imageSize(2)-1)/2;
        cy = (imageSize(1)-1)/2;
        [fx, fy] = vision.internal.calibration.computeFocalLength(homographies, cx, cy);
        A = vision.internal.calibration.constructIntrinsicMatrix(fx, fy, cx, cy, 0);
        if ~isreal(A)
            error(message('vision:calibrate:complexCameraMatrix'));
        end
    else        
        V = computeV(homographies);
        B = computeB(V);
        A = computeIntrinsics(B);
    end
% else
%     % initial guess for the intrinsics has been provided. No need to solve.
%     A = initIntrinsics';
% end


%% % Compute translation and rotation vectors for all images

% [rvecs, tvecs] = computeExtrinsics(A, H);
numImages = size(homographies, 3);
rotationVectors = zeros(3, numImages);
translationVectors = zeros(3, numImages); 
Ainv = inv(A);
for i = 1:numImages
    H = homographies(:, :, i);
    h1 = H(:, 1);
    h2 = H(:, 2);
    h3 = H(:, 3);
    lambda = 1 / norm(Ainv * h1); %#ok
    
    % 3D rotation matrix
    r1 = lambda * Ainv * h1; %#ok
    r2 = lambda * Ainv * h2; %#ok
    r3 = cross(r1, r2);
    R = [r1,r2,r3];
    
    rotationVectors(:, i) = vision.internal.calibration.rodriguesMatrixToVector(R);
    
    % translation vector
    t = lambda * Ainv * h3;  %#ok
    translationVectors(:, i) = t;
end

rotationVectors = rotationVectors';
translationVectors = translationVectors';
% 
% if isempty(initRadial)
%     radialCoeffs = zeros(1, cameraModel.NumRadialDistortionCoefficients);
% else
%     radialCoeffs = initRadial;
% end

worldUnits = 'mm';
cameraModel.EstimateSkew = false;
cameraModel.NumRadialDistortionCoefficients = 2;
cameraModel.EstimateTangentialDistortion = false;
radialCoeffs = zeros(1, cameraModel.NumRadialDistortionCoefficients);
calibrationParams.shouldComputeErrors = true;
initialParams = cameraParameters('IntrinsicMatrix', A', ...
    'RotationVectors', rotationVectors, ...
    'TranslationVectors', translationVectors, 'WorldPoints', worldPoints, ...
    'WorldUnits', worldUnits, 'EstimateSkew', cameraModel.EstimateSkew,...
    'NumRadialDistortionCoefficients', cameraModel.NumRadialDistortionCoefficients,...
    'EstimateTangentialDistortion', cameraModel.EstimateTangentialDistortion,...
    'RadialDistortion', radialCoeffs, 'ImageSize', imageSize);
fprintf('RD1:');
disp(initialParams.RadialDistortion);
fprintf('A1:');
disp(initialParams.IntrinsicMatrix);
%% refine the initial estimate and compute distortion coefficients using non-linear least squares minimization 
errors = refine(initialParams, imagePoints, calibrationParams.shouldComputeErrors);
fprintf('RD2:');
disp(initialParams.RadialDistortion);
fprintf('A2:');
disp(initialParams.IntrinsicMatrix);
%% present result
  % Visualize calibration accuracy.
  figure;
  showReprojectionErrors(initialParams);

  % Visualize camera extrinsics.
  figure;
  showExtrinsics(initialParams);
  drawnow;

  % Plot detected and reprojected points.
  figure; 
  imshow(I); 
  hold on
  plot(imagePoints(:, 1, 1), imagePoints(:, 2, 1), 'go');
  plot(initialParams.ReprojectedPoints(:, 1, 1), initialParams.ReprojectedPoints(:, 2, 1), 'r+');
  legend('Detected Points', 'ReprojectedPoints');
  hold off



  %%

function H = computeHomography(imagePoints, worldPoints)
% Compute projective transformation from worldPoints to imagePoints

validPointsIdx = ~isnan(imagePoints(:,1));

H = fitgeotrans(worldPoints(validPointsIdx,:), imagePoints(validPointsIdx,:), 'projective');
H = (H.T)';
H = H / H(3,3);
end

function V = computeV(homographies)
% Vb = 0

numImages = size(homographies, 3);
V = zeros(2 * numImages, 6);
for i = 1:numImages
    H = homographies(:, :, i)';
    V(i*2-1,:) = computeLittleV(H, 1, 2);
    V(i*2, :) = computeLittleV(H, 1, 1) - computeLittleV(H, 2, 2);
end

end
%--------------------------------------------------------------------------
function v = computeLittleV(H, i, j)
    v = [H(i,1)*H(j,1), H(i,1)*H(j,2)+H(i,2)*H(j,1), H(i,2)*H(j,2),...
         H(i,3)*H(j,1)+H(i,1)*H(j,3), H(i,3)*H(j,2)+H(i,2)*H(j,3), H(i,3)*H(j,3)];
end

%--------------------------------------------------------------------------     
function B = computeB(V)
% lambda * B = inv(A)' * inv(A), where A is the intrinsic matrix

[~, ~, U] = svd(V);
b = U(:, end);

% b = [B11, B12, B22, B13, B23, B33]
B = [b(1), b(2), b(4); b(2), b(3), b(5); b(4), b(5), b(6)];
end
%--------------------------------------------------------------------------
function A = computeIntrinsics(B)
% Compute the intrinsic matrix

cy = (B(1,2)*B(1,3) - B(1,1)*B(2,3)) / (B(1,1)*B(2,2)-B(1,2)^2);
lambda = B(3,3) - (B(1,3)^2 + cy * (B(1,2)*B(1,3) - B(1,1)*B(2,3))) / B(1,1);
fx = sqrt(lambda / B(1,1));
fy = sqrt(lambda * B(1,1) / (B(1,1) * B(2,2) - B(1,2)^2));
skew = -B(1,2) * fx^2 * fy / lambda;
cx = skew * cy / fx - B(1,3) * fx^2 / lambda;
A = vision.internal.calibration.constructIntrinsicMatrix(fx, fy, cx, cy, skew);
if ~isreal(A)
    error(message('vision:calibrate:complexCameraMatrix'));
end

end
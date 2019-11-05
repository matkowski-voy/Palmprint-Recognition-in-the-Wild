% demo script by voy

%% prepare data for ROI_LAnet training
fprintf('augment images, prepare and save imdb for training\n')

run img_mask_transform_buildImdb

fprintf('press ket to continue\n')
pause

%% train ROI-LAnet
fprintf('train ROI-LAnet\n')

run demo2

fprintf('press key to continue\n')
pause

%% show some demo results (detected landmarks and ROIs)
fprintf('show landmarks and ROIs\n')

run demo3

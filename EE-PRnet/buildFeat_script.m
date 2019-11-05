% buildFeat script

expNum = 'exp1';
imgType = 'img';
netName = 'net-epoch-10.mat';

netPath = fullfile(pwd,'results',expNum,netName);
paths.imagesTrain = fullfile('../databases',dbName,'/flip/',imgType,'/train/');
paths.imagesTest = fullfile('../databases',dbName,'/flip/',imgType,'/test/');

fprintf('building gallery set features\n')
buildFeat(expNum,imgType,'train',paths.imagesTrain,netPath)
fprintf('building probe set features\n')
buildFeat(expNum,imgType,'test',paths.imagesTest,netPath)
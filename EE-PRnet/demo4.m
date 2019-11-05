% demo3 by voy build feat using pretrained network and build pls models 
% and demo search showin rank ordered candidates from the gallery
% example of simple search
close all; clc;

run masksetup

imgType = 'imgOrg';
dbName = 'NTU-test';
dataListName = 'DataNclass.mat';
expNum = 'exp1';


%% setup paths to network
netName = 'EE_PRnet-epoch-60.mat'; % file with saved network after training
netPath = fullfile('../preTrainedNetworks',netName);
%% build feature sets
fprintf('building gallery set features\n')
galleryType = 'train';
path2gallery = fullfile('../databases',dbName,'/flip/',imgType,'/train/');
tic
% buildFeat(expNum,imgType,galleryType,path2gallery,netPath)
toc
tic
fprintf('building probe set features\n')
queryType = 'test';
path2probe = fullfile('../databases',dbName,'/flip/',imgType,queryType);
% buildFeat(expNum,imgType,queryType,path2probe,netPath)
toc


%% build PLS models
fprintf('building PLS models\n')
kParam = 50;
buildPLSmodels(expNum,netName,kParam)


%% evaluate
fprintf('probe set evaluation: 1 to many comparison\n')
evaluate(expNum,'test','pls')
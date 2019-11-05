% demo3 by voy build feat using pretrained network and build pls models 
% and demo search showin rank ordered candidates from the gallery
% example of simple search
close all; clc;

run masksetup

imgType = 'img';
dbName = 'NTU-PI-demo';
dataListName = 'DataNclass.mat';
expNum = 'exp1';


%% setup paths to network
netName = 'net-epoch-10.mat'; % file with saved network after training
netPath = fullfile(pwd,'results',expNum,netName);
%% build feature sets
fprintf('building gallery set features\n')
galleryType = 'train';
path2gallery = fullfile('../databases',dbName,'/flip/',imgType,'/train/');
tic
buildFeat(expNum,imgType,galleryType,path2gallery,netPath)
toc
tic
fprintf('building probe set features\n')
queryType = 'evidence';
path2evidence = fullfile('../databases',dbName,'/flip/',imgType,queryType);
buildFeat(expNum,imgType,queryType,path2evidence,netPath)
toc


%% build PLS models
fprintf('building PLS models\n')
kParam = 50;
buildPLSmodels(expNum,netName,kParam)


%% search
imgStruct=idxImages(path2evidence,path2gallery);
clistRange = 3; % range of the candidate's list
fprintf('search IDs in the gallery\n')
simpleSearchPreview(expNum,queryType,imgStruct,'pls',clistRange)
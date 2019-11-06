% demo2 by voy build imdb, train EE-PRnet and build feature sets build pls 
% models, do recognition and plot cmc
close all; clc;

run masksetup

imgType = 'img';
dbName = 'NTU-PI-demo';
dataListName = 'DataNclass.mat';
expNum = 'exp1';


%% uncomment to create imdb
fprintf('building IMDB\n');
tic
augment_build_imdb(expNum,dbName,imgType,dataListName);
toc
%% 

%% setup paths to data
paths.imagesTrain = fullfile('../databases',dbName,'/flip/',imgType,'/train/');
paths.imagesTest = fullfile('../databases',dbName,'/flip/',imgType,'/test/');
% paths.ROI_LAnet_name = fullfile('../ROI-LAnet/results/exp1/net-epoch-10.mat');
%% set path to alignment network 
paths.ROI_LAnet_name = 'AlignNet-epoch-25.mat';
paths.results = fullfile(pwd,'results');


%% load training parameters
load(fullfile(pwd,'trainParams',expNum,'trainParams.mat'));

%% train network
topts.gpus=1;
% trainEE_PRnet(paths,topts,imgType,expNum);


%% setup paths to network
netName = 'net-epoch-40.mat'; % file with saved network after training
netPath = fullfile(pwd,'results',expNum,netName);
%% build feature sets
fprintf('building gallery set features\n')
tic
buildFeat(expNum,imgType,'train',paths.imagesTrain,netPath)
toc
tic
fprintf('building probe set features\n')
buildFeat(expNum,imgType,'test',paths.imagesTest,netPath)
toc


%% build PLS models
fprintf('building PLS models\n')
kParam = 25;
buildPLSmodels(expNum,netName,kParam)


%% evaluate
fprintf('probe set evaluation 1 to many comparison\n')
evaluate(expNum,'test','pls')
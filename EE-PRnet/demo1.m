% demo1 by voy build imdb and train EE-PRnet
close all; clc;

run masksetup

imgType = 'imgOrg';
dbName = 'NTU-test';
dataListName = 'DataNclass.mat';
expNum = 'exp1';

%% comment if imdb already created
fprintf('CREATING IMDB\n');
tic
augment_build_imdb(expNum,dbName,imgType,dataListName);
toc
%% 

%% set up paths to data
paths.imagesTrain = fullfile('../databases',dbName,'/flip/',imgType,'/train/');
paths.imagesTest = fullfile('../databases',dbName,'/flip/',imgType,'/test/');
%% set path to alignment network 
% paths.ROI_LAnet_name = fullfile('../ROI-LAnet/results/exp1/net-epoch-10.mat');
paths.ROI_LAnet_name = 'AlignNet-epoch-25.mat';
paths.results = fullfile(pwd,'results');

%% load training parameters
load(fullfile(pwd,'trainParams',expNum,'trainParams.mat'));


%% train network
topts.gpus=1;
trainEE_PRnet(paths,topts,imgType,expNum); 
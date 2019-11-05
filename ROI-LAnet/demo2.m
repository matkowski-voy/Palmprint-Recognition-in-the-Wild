%% train the alignment network ROI-LAnet
close all; clc;

run masksetup

imgType = 'imgGen';
dbName = 'NTU-PI-demo';
dataListName = 'DataAlign.mat';
expNum = 'exp1';

paths.imagesTrain = fullfile('../databases',dbName,'/flip/',imgType,'/');
paths.results = fullfile(pwd,'results');
paths.list = fullfile(pwd,'imdbDataList',dataListName);
load(fullfile(pwd,'trainParams',expNum,'trainParams.mat'));

trainROI_LAnet(paths,topts,expNum);


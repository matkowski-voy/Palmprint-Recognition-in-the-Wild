function []=trainEE_PRnet(paths,usertopts,imgType,expNum)
% path: struct specifying location of data and results
% usertopts: struct with training options provided by user 
% mode: 1 train model  (will initialize new model)
%       2 evaluate     (will load existing model)

%% Define default training parameters
topts=struct();
topts.lr=1e-3;
topts.numEpochs = 10;
topts.batchSize=128;
topts.gpus=1;
topts.saveResults=1;
topts.weightDecay = 0 ;
topts.momentum = 0.9 ;
topts.f_sROI_layer = 'pool3';
topts.fineTuneDepth = 4;
topts.ROIsize = [112 112];
topts.embedding = 1;
topts.embeddingDim = 512;
topts.drop3 = 0.5;
topts.drop4 = 0.5;
topts.strategy = 'S1';
topts.lr_factor = 1;
topts.drop_off_epoch = 35;
topts.ft_align_epoch = 20;
topts.at_augment_on = 40;
%% Load user training options

% overwrite default parameters with user provided parameters (if any)
if nargin>=2
    topts = vl_argparse(topts, usertopts);
end

display('Training params:');
topts

% set random seed 
seed=2018;
rng(seed)

%% Load training data

Data = load(fullfile(pwd,'imdbDataList',expNum,'DataNclass.mat'));

% concat tr and val data (for passing to training algorithm)
Ntr = length(Data.imdb.tr.image);
Nval = length(Data.imdb.val.image);

image=[strcat(paths.imagesTrain,Data.imdb.tr.image)';...
    strcat(paths.imagesTest,Data.imdb.val.image)'];
label=[Data.imdb.tr.label Data.imdb.val.label];

netopts = topts;
netopts.numClasses = Data.imdb.meta.numClassesGallery;
netopts.imgType = imgType;
netopts.ROI_LAnet_name = paths.ROI_LAnet_name;

% create net
if(strcmp(imgType,'img') || strcmp(imgType,'ROI') || ...
        strcmp(imgType,'imgOrg'))
fprintf('Creating end to end palmprint recognition network EE-PRnet\n');

net = createEE_PRnet(netopts);
end
% 
% set meta spec
net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:Data.imdb.meta.numClassesGallery,'UniformOutput',false);
net.meta.inuptType.name = imgType;

mkdir(fullfile(paths.results,expNum));
save(fullfile(paths.results,expNum,'topts.mat'),'topts');


%% Train 
imdb.image = image;
imdb.label = label;
imdb.setTransformation = Data.imdb.meta.setTransformation;
opts=struct();
opts.weightDecay = topts.weightDecay;
opts.momentum = topts.momentum;
opts.batchSize = topts.batchSize;
opts.gpus = topts.gpus;
opts.expDir = fullfile(paths.results,expNum,'/');
opts.train = 1:Ntr;
opts.val = Ntr+1:Ntr+Nval;
opts.numEpochs = topts.numEpochs;
opts.derOutputs = {'objective',1};
opts.learningRate = topts.lr;
opts.strategy = topts.strategy;
opts.lr_factor = topts.lr_factor;
opts.drop_off_epoch = topts.drop_off_epoch;
opts.ft_align_epoch = topts.ft_align_epoch;
imdb.epoch_cnt = 1;
 
if(strcmp(imgType,'img') || strcmp(imgType,'imgOrg'))
getBatch = @(imdb,batchIdx) getBatchNclassSpecial(imdb,batchIdx,Ntr+1,topts.at_augment_on);
else
getBatch = @(imdb,batchIdx) getBatchNclass(imdb,batchIdx,Ntr+1);
end

% train
cnn_train_dagEE_PRnet(net, imdb, getBatch, opts);


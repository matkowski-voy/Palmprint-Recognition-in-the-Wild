function result=trainROI_LAnet(paths,usertopts,expNum)

% Define default training parameters
topts=struct();
topts.lr=1e-3;
topts.numEpochs = 25;
topts.batchSize=128;
topts.gpus=1;
topts.saveResults=1;
topts.weightDecay = 0 ;
topts.momentum = 0.9 ;
topts.f_shand_layer = 'pool3';
topts.ft_epoch = 10;

if nargin>=2
    topts = vl_argparse(topts, usertopts);
end

seed=2018;
rng(seed)
fprintf('Creating alignemnt network ROI-LAnet\n');
[net, prune_idx] = createROI_LAnet(topts.f_shand_layer);


Data = load(paths.list);
Ntr = length(Data.imdb.tr.image);
Nval = length(Data.imdb.val.image);
images=[strcat(paths.imagesTrain,Data.imdb.tr.image)';...
    strcat(paths.imagesTrain,Data.imdb.val.image)'];
label=cat(4,Data.imdb.tr.label, Data.imdb.val.label);

imdb.image = images;
imdb.label = label;
opts=struct();
opts.weightDecay = topts.weightDecay;
opts.momentum = topts.momentum;
opts.batchSize = topts.batchSize;
opts.gpus = topts.gpus;
opts.expDir = fullfile(paths.results,expNum,'/');
opts.train = 1:Ntr;
opts.val = Ntr+1:Ntr+Nval;
opts.numEpochs = topts.numEpochs;
opts.learningRate = topts.lr;
opts.derOutputs = {'loss',1};    
opts.ft_epoch = topts.ft_epoch;
opts.prune_idx = prune_idx;

getBatch = @(imdb,batchIdx) getBatchROI_LAnet(imdb,batchIdx);

cnn_train_dag_ROI_LAnet(net, imdb, getBatch, opts);

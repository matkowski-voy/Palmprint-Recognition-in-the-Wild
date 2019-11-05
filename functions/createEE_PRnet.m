function net = createEE_PRnet(varargin)

% default options
opts=struct();
opts.lr=1e-3;
opts.numEpochs = 10;
opts.batchSize=64;
opts.gpus=1;
opts.saveResults=1;
opts.weightDecay = 0 ;
opts.momentum = 0.9 ;
opts.f_sROI_layer = 'pool3';
opts.numClasses = 2035;
opts.fineTuneDepth = 4;
opts.ROIsize = [112 112];
opts.embeddingDim = 512;
opts.embedding = 1;
opts.drop3 = 0.5;
opts.drop4 = 0.5;
opts.imgType = 'img';
opts.ROI_LAnet_name = 'AlignNet-epoch-25.mat';
opts.strategy = 'S4';
opts.lr_factor = 1;
opts.drop_off_epoch = 35;
opts.ft_align_epoch = 20;
opts.at_augment_on = 40;

% load user-specified options
opts = vl_argparse(opts, varargin);

%% Create vgg-16 base model
    net = dagnn.DagNN.fromSimpleNN(load('imagenet-vgg-verydeep-16.mat'));
    Nlayers = net.getLayerIndex(opts.f_sROI_layer);
    while length(net.layers)>Nlayers
        net.removeLayer(net.layers(end).name);
    end
    netStruct = net.saveobj;
    netStruct.vars(1).name='Input';
    netStruct.layers(1).inputs{1}='Input';    

    net = dagnn.DagNN.loadobj(netStruct);   
    fprintf('pre-trained VGG-16 module loaded\n');
    fprintf(strcat('pruned at: ',opts.f_sROI_layer,'\n'));

 
if(strcmp(opts.imgType,'img') || strcmp(opts.imgType,'imgOrg'))

    net = connectNetModules(net,opts.ROI_LAnet_name,'theta',opts.ROIsize);
    fprintf('Alingment and Recognition Modules connected\n');
    fprintf(strcat('Alignment Module filename: ',opts.ROI_LAnet_name,'\n'));
   
end

net.addLayer('f_sROI', dagnn.LRN('param', [512 1e-6 1 0.5]),...  
net.layers(end).outputs{1}, {'f_sROIout'}, {}) ;


%% init embedding
if(opts.embedding == true)    
    if(opts.ROIsize(1) == 112)
        opts.conv1size = 14;
    end
    if(strcmp(opts.f_sROI_layer,'pool3'))
        opts.conv1depth = 256; 
    end
    opts.conv1filters = opts.embeddingDim;
    net = initEmbedding(net,opts);
end
fprintf('Embedding and Top Layers initialized\n');
    

% simple setup learing rates
for i=1:length(net.params)
    
    net.params(i).learningRate = 1;
    if(length(net.params) - i < opts.fineTuneDepth)
         net.params(i).learningRate = 1;
         fprintf('trainable: %s\n',net.params(i).name)
    else
         net.params(i).learningRate = 0;
         fprintf('frozen: %s\n',net.params(i).name)
    end
    

end

if(strcmp(opts.strategy,'S1')) % freeze dropouts for S1 strategy
    net.layers(21).block.rate = 0;
    net.layers(24).block.rate = 0;
end
if(strcmp(opts.strategy,'S3')) % freeze dropouts for S3 strategy
    net.layers(21).block.rate = 0;
    net.layers(24).block.rate = 0;
end


lossBlock = dagnn.Loss('loss','softmaxlog');
fprintf('Softmax Loss Layer added\n')   
errorBlock = dagnn.Loss('loss','classerror');
net.addLayer('loss', lossBlock, {'pred','label'},'objective') ;
net.addLayer('top1err', errorBlock,{'pred','label'},'error');
if(opts.numClasses >= 30)
net.addLayer('top30err',dagnn.Loss('loss','topkerror','opts',{'topk',30}),{'pred','label'},'top30err');   
end    

net.conserveMemory = true; % false: store all the intermediate results
net.vars(end).precious=1;
end

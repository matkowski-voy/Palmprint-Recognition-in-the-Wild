function [net, prune_idx] = createROI_LAnet(f_shand_layer)

    % prune pretrained VGG-16 at f_sROI_layer (pool3)
    net = dagnn.DagNN.fromSimpleNN(load('imagenet-vgg-verydeep-16.mat')); 
    Nlayers = net.getLayerIndex(f_shand_layer);
    while length(net.layers)>Nlayers
        net.removeLayer(net.layers(end).name);
    end

    netStruct = net.saveobj;
    netStruct.vars(1).name='Input';
    netStruct.layers(1).inputs{1}='Input';
    netStruct = netNamePrefix(netStruct,'AN','AN','AN'); 
    net = dagnn.DagNN.loadobj(netStruct);
    net.addLayer('f_sHand', dagnn.LRN('param', [512*2 1e-6 1 0.5]),{net.layers(end).outputs{1}}, {'f_sHandout'}, {}) ;

    nL1 = size(net.params);
    
    % add regresion network with fully conected layers and dropouts
    net = addTopRegNet(net);
    
    nL2 = size(net.params);
    
    % add loss function
    lossBlock = dagnn.PDist('p',2,'noRoot',true,'aggregate',true);
    net.addLayer('loss', lossBlock, {'theta', 'thetaGt'}, {'loss'},{}) ;
    
    % set trainable and frozen parameters  (see: nL2(2)-nL1(2))
    prune_idx = nL1(2);
    for i=1:length(net.params)
        if(length(net.params)-i >= (nL2(2)-nL1(2)))
            net.params(i).learningRate = 0;
             fprintf('frozen: %s\n',net.params(i).name)
        else
            net.params(i).learningRate = 1;
            fprintf('trainable: %s\n',net.params(i).name)
        end
    end

net.conserveMemory = true; 
net.vars(end).precious=1;
end

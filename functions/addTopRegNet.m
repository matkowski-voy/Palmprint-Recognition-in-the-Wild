function [net] = addTopRegNet(net)

gaussianStd = 0.001;

%% reg layer 1
conv1depth = 256;
opts.conv1size = 7;
opts.conv1filters = 512;
sizeconv1 = [opts.conv1size opts.conv1size conv1depth opts.conv1filters];
convBlock = dagnn.Conv('size', sizeconv1, 'hasBias', true);
net.addLayer('fc1', convBlock, {net.layers(end).outputs{1}}, {'fc1out'}, {'fc1f', 'fc1b'});
net.params(net.layers(net.getLayerIndex('fc1')).paramIndexes(1)).value = randn(sizeconv1,'single')*gaussianStd;
net.params(net.layers(net.getLayerIndex('fc1')).paramIndexes(2)).value = randn(sizeconv1(end),1,'single')*gaussianStd;


net.addLayer('relu1', dagnn.ReLU('leak',0.01), {net.layers(net.getLayerIndex('fc1')).outputs{1}}, {'relu1out'}, {});


%%  reg layer 2
opts.conv2filters=128;
opts.conv2size = 1;
sizeconv2 = [opts.conv2size opts.conv2size opts.conv1filters opts.conv2filters];
convBlock2 = dagnn.Conv('size', sizeconv2, 'hasBias', true) ;
net.addLayer('drop1',dagnn.DropOut('rate',0.2),{net.layers(end).outputs{1}},{'drop1out'},{});
net.addLayer('fc2', convBlock2, {net.layers(end).outputs{1}}, {'fc2out'}, {'fc22f', 'fc22b'});
net.params(net.layers(net.getLayerIndex('fc2')).paramIndexes(1)).value = randn(sizeconv2,'single')*gaussianStd;
net.params(net.layers(net.getLayerIndex('fc2')).paramIndexes(2)).value = randn(sizeconv2(end),1,'single')*gaussianStd;


 net.addLayer('relu2', dagnn.ReLU('leak',0.01), {net.layers(end).outputs{1}}, {'relu2out'}, {});

% 
%%  reg layer 3
opts.conv3filters=10;
opts.conv3filters=18;
opts.conv3size = 1;
sizeconv3 = [opts.conv3size opts.conv3size opts.conv2filters opts.conv3filters];
convBlock2 = dagnn.Conv('size', sizeconv3, 'hasBias', true) ;
net.addLayer('drop2',dagnn.DropOut('rate',0.1),{net.layers(end).outputs{1}},{'drop2out'},{});
net.addLayer('fc3', convBlock2, {net.layers(end).outputs{1}}, {'theta'}, {'fc3f', 'fc3b'});
net.params(net.layers(net.getLayerIndex('fc3')).paramIndexes(1)).value = randn(sizeconv3,'single')*gaussianStd;
net.params(net.layers(net.getLayerIndex('fc3')).paramIndexes(2)).value = randn(sizeconv3(end),1,'single')*gaussianStd;






function net = initEmbedding(net,opts)

g = 0.001; % normal oroginal value

  sizeconv1 = [opts.conv1size opts.conv1size opts.conv1depth opts.conv1filters];

    convBlock = dagnn.Conv('size', sizeconv1, 'hasBias', true,'stride',1);
    net.addLayer('drop0R',dagnn.DropOut('rate',opts.drop3),{net.layers(end).outputs{1}},{'drop0outR'},{});
    net.addLayer('fcr1_4R', convBlock, {net.layers(end).outputs{1}}, {'fc1R'}, {'fc1fR', 'fc1bR'});
    net.params(net.layers(net.getLayerIndex('fcr1_4R')).paramIndexes(1)).value = randn(sizeconv1,'single')*g;
    net.params(net.layers(net.getLayerIndex('fcr1_4R')).paramIndexes(2)).value = randn(sizeconv1(end),1,'single')*g;    

    sizeconv2 = [1 1 opts.conv1filters opts.numClasses];

    convBlock = dagnn.Conv('size', sizeconv2, 'hasBias', true,'stride',1);
    net.addLayer('drop1R',dagnn.DropOut('rate',opts.drop4),{net.layers(end).outputs{1}},{'drop1outR'},{});
    net.addLayer('fcr2_4R', convBlock, {net.layers(end).outputs{1}}, {'pred'}, {'fc2fR', 'fc2bR'});
    net.params(net.layers(net.getLayerIndex('fcr2_4R')).paramIndexes(1)).value = randn(sizeconv2,'single')*g;
    net.params(net.layers(net.getLayerIndex('fcr2_4R')).paramIndexes(2)).value = randn(sizeconv2(end),1,'single')*g;


    
end
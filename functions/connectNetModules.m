function net = connectNetModules(net,ROI_LAnet_name,transformation,ROIsize)

ROI_LAnet = load(ROI_LAnet_name) ;
ROI_LAnet = ROI_LAnet.net ;
ROI_LAnet = dagnn.DagNN.loadobj(ROI_LAnet);
ROI_LAnet.removeLayer(ROI_LAnet.layers(end).name);

G = dagnnExtra.TpsGridGenerator('Ho',ROIsize(1),'Wo',ROIsize(2),'k',3,'lambda',0,'useGPU',1);
ROI_LAnet.addLayer('grid_generator',G,{transformation},{'grid'});
Sampler = dagnn.BilinearSampler();

ROI_LAnet.addLayer('sampler',Sampler,{'Input224','grid'},{'Input'});
netStruct = net.saveobj;
netStruct2  = ROI_LAnet.saveobj;


netStructCombo.vars = [netStruct2.vars netStruct.vars];
netStructCombo.params = [netStruct2.params netStruct.params];
netStructCombo.layers = [netStruct2.layers netStruct.layers];
netStructCombo.meta = [netStruct.meta];

net = dagnn.DagNN.loadobj(netStructCombo);

end
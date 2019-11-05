% demo script by voy matkowski
% visualize hand landmarks and palmprint region of interest (ROI) 
% extracted by ROI-LAnet 
close all; clc;

dbName = 'NTU-PI-demo'
imgSet = 'test'
netName = 'net-epoch-25.mat'
% netName = 'AlignNet-epoch-25.mat'
expNum = 'exp1'
h = 56;
w = 56;
h_ROI = 112;
w_ROI = 112;

ROI_LAnet = load(fullfile(pwd,'results',expNum,netName)) ;
% ROI_LAnet = load(fullfile('../preTrainedNetworks',netName));
ROI_LAnet = ROI_LAnet.net ;
ROI_LAnet = dagnn.DagNN.loadobj(ROI_LAnet);
ROI_LAnet.removeLayer(ROI_LAnet.layers(end).name) ;
ROI_LAnet.vars(ROI_LAnet.getVarIndex('theta')).precious=1 ;
ROI_LAnet.mode = 'test';

pathImg = fullfile('../databases',dbName,'/flip/img/',imgSet);
file = dir(pathImg); file = file(3:end);

for i=1:length(file)

I = imread(fullfile(pathImg,file(i).name));
I_resized = vl_imreadjpeg({fullfile(pathImg,file(i).name)},'resize',[h w],'SubtractAverage',[123.68 116.78 103.94]);
I_resized = I_resized{1};
ROI_LAnet.eval({'Input',I_resized}) ;

theta_hat = ROI_LAnet.vars(ROI_LAnet.getVarIndex('theta')).value;
checkTps1 = squeeze(theta_hat);

L=length(theta_hat);
G = dagnnExtra.TpsGridGenerator('Ho',h_ROI,'Wo',w_ROI,'k',sqrt(L/2),'lambda',0) ;
Sampler = dagnn.BilinearSampler ;   
T_G = G.forward({theta_hat}) ;
I_ROI = Sampler.forward({single(I),T_G{1}}) ;
I_ROI = I_ROI{1};

points = squeeze(theta_hat);
maxA = size(I,1);
minA = 1;
pNew = (maxA-minA)/(1 + 1).*(points-1) + maxA;
figure;imshow(I); hold on;
plot(pNew(1:9),pNew(10:18),'.r');

figure; imshow(uint8(I_ROI));

fprintf('press key to continue\n')
pause  
close all
end




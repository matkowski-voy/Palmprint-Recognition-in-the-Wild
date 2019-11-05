% build features
function []=buildFeat(expNum,imgType,imgSet,imgPath,netPath)

% imgType = 'img';
% dbName = 'NTU-PI-demo';
% expNum = 'exp1';
% netName = 'net-epoch-10.mat';
% imgSet = 'test';
% imgPath = fullfile('../databases/',dbName,'/flip/',imgType,imgSet); % probe images
% netPath = fullfile(pwd,'results',expNum,netName);

featLayerName = 'fc1R';
featDim =512;

%%load network
netNclass = load(netPath);
netNclass = dagnn.DagNN.loadobj(netNclass.net);
netNclass.vars(netNclass.getVarIndex(featLayerName)).precious=1 ;

netNclass.move('gpu');
netNclass.meta.mode = 'test';
netNclass.mode = 'test';

imsize224 = [224 224]; % input image size
imsize = [112 112];

if(strcmp(imgType,'img') || strcmp(imgType,'imgOrg'))
     imsize = imsize/2;
end
avimnet = [123.68 116.78 103.94];

mkdir(fullfile(pwd,'results',expNum,'features/',imgSet));
pathSaveFeat = fullfile(pwd,'results',expNum,'features/',imgSet);
newFileTest = dir(imgPath); newFileTest = newFileTest(3:end);

Vec = zeros(length(newFileTest),featDim+3);
ind =0;
indVec = 1;

for i=1:length(newFileTest)
    ind = ind +1;
%     fprintf('iter: %d\n',i);
     testImg = newFileTest(ind).name;
   
    I = vl_imreadjpeg({strcat(imgPath,'/',testImg)},'resize',imsize,'SubtractAverage',avimnet,'Pack');
    I = I{1};
    I = gpuArray(I);
    if(strcmp(imgType,'img'))
        I_224 = vl_imreadjpeg({strcat(imgPath,'/',testImg)},'resize',imsize224,'SubtractAverage',avimnet,'Pack');
        I_224 = I_224{1};
        I_224 = gpuArray(I_224);
    end
    if(strcmp(imgType,'imgOrg'))
        I_224 = vl_imreadjpeg({strcat(imgPath,'/',testImg)},'SubtractAverage',avimnet);
        I_224 = I_224{1};
        I_224 = gpuArray(I_224);
    end
    
    
    c1 = strfind(testImg,'Class');
    c2 = strfind(testImg,'.');
    classId(i) = str2num(testImg(c1+5:c2(1)-1));

    if(strcmp(imgType,'img') || strcmp(imgType,'imgOrg'))
        netNclass.eval({'ANInput',I,'Input224',I_224});
    else
         netNclass.eval({'Input',I});
    end

    vec = gather(squeeze(netNclass.vars(netNclass.getVarIndex(featLayerName)).value));
    vec = reshape(vec,size(vec,1)*size(vec,2)*size(vec,3),1)';
    Vec(indVec,:) = [vec  classId(i) 0 i];
    name{indVec} = testImg;
    indVec = indVec + 1;
    
end
save(fullfile(pathSaveFeat,'set.mat'),'Vec','name','-v7.3');
fprintf('features saved\n');
end





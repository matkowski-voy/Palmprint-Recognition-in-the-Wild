% recognition
function[]=simpleSearchPreview(expNum,featureSet,imgStruct,models,listRange)
% expNum = 'exp1';
% imgSet = 'test';
% models = 'pls';
pathSet = fullfile(pwd,'results',expNum,'features',featureSet);
pathModels = fullfile(pwd,'results',expNum,models);
paths.results = fullfile(pwd,'results');

model_ova = dir(pathModels); model_ova = model_ova(3:end);
A = importdata(fullfile(pathSet,'set.mat'));
A = A.Vec;
A = gpuArray(A);

    numModels = length(model_ova);
    descritpionNum = 3;
    numFeatures = size(A,2)-descritpionNum; % -3 because last 3 cols is a description part
    beta = zeros(numFeatures,numModels);
    x_mu = zeros(numFeatures,numModels);
    y_mu = zeros(1,numModels);
    x_sigma = zeros(numFeatures,numModels);
    y_sigma = zeros(1,numModels);
    DictClass = zeros(1,numModels);

        beta = gpuArray(beta);
        x_mu = gpuArray(x_mu);
        y_mu = gpuArray(y_mu);
        x_sigma = gpuArray(x_sigma);
        y_sigma = gpuArray(y_sigma);
    
    for i=1:numModels
        
        load(fullfile(pathModels,model_ova(i).name))
        beta(:,i) = b;
        x_mu(:,i) = xMu;
        xSigma(xSigma == 0) = 1;
        x_sigma(:,i) = xSigma;
        y_mu(i) = yMu;
        y_sigma(i) = ySigma; 
       c1 = strfind(model_ova(i).name,'_');
       DictClass(i) = str2double(model_ova(i).name(11:c1-1));
       
    end


data = A(:,1:end-descritpionNum);
DictData = A(:,end-2);

tic
s = cell(size(data,1),1);
respVec = zeros(size(data,1),numModels);
respVec = gpuArray(respVec);

img = imread(fullfile(imgStruct.path2evidence,imgStruct.input_name));
figure; imshow(img); 
title('probe-evidence img');
for kk=1:size(data,1)
    
    data_in = data(kk,:);
    dataTest = (data_in-xMu)./xSigma;    
    response = beta'*dataTest';
    respVec(kk,:) = response;
    [~,ind_r] = sort(response,'descend');
    candidate_list=DictClass(ind_r);
    
    for jj=1:listRange
        cIdx=find(imgStruct.classId == candidate_list(jj));
        for cc=1:length(cIdx)
            img = imread(fullfile(imgStruct.path2gallery,imgStruct.image_name{cIdx(cc)}));
            subplot(1,length(cIdx),cc); imshow(img)
%             figure; imshow(img); 
            if(cc==1)
            title(strcat('rank-',num2str(jj),'-candidate ID',num2str(candidate_list(jj))));
            end
        end
        fprintf('press key to see the next candidate\n')
        pause
    end
    
end
toc
end
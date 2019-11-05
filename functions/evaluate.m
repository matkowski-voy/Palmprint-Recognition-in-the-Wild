% recognition
function[]=evaluate(expNum,imgSet,models)
% expNum = 'exp1';
% imgSet = 'test';
% models = 'pls';
pathSet = fullfile(pwd,'results',expNum,'features',imgSet);
pathModels = fullfile(pwd,'results',expNum,models);
paths.results = fullfile(pwd,'results');

model_ova = dir(pathModels); model_ova = model_ova(3:end);
A = importdata(fullfile(pathSet,'set.mat'));
A = A.Vec;
A = gpuArray(A);

    numModels = length(model_ova);
    descritpionNum = 3;
    numFeatures = size(A,2)-descritpionNum; % -3 because last 3 cols is a description part
    Beta = zeros(numFeatures,numModels);
    x_mu = zeros(numFeatures,numModels);
    y_mu = zeros(1,numModels);
    x_sigma = zeros(numFeatures,numModels);
    y_sigma = zeros(1,numModels);
    DictClass = zeros(1,numModels);

        Beta = gpuArray(Beta);
        x_mu = gpuArray(x_mu);
        y_mu = gpuArray(y_mu);
        x_sigma = gpuArray(x_sigma);
        y_sigma = gpuArray(y_sigma);
    
    for i=1:numModels
        
        load(fullfile(pathModels,model_ova(i).name))
        Beta(:,i) = b;
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

for kk=1:size(data,1)
    
    data_in = data(kk,:);
    dataTest = (data_in-xMu)./xSigma;    
    response = Beta'*dataTest';
    respVec(kk,:) = response;
    [~,ind_r] = sort(response,'descend');
    s{kk} = find(DictData(kk) == unique(DictClass(ind_r),'stable'));
    
end
toc

rankPP = zeros(1,size(data,1));
% case only if non-unique classes can be retrieved
for ii=1:size(data,1)
    z = s{ii};
    if(numel(z) == 0)
        z = 0;
    end
    rankPP(ii) = gather(z(1));
end

rankPP = rankPP(rankPP ~= 0);
for i =1:length(rankPP)
    cmcTab(i) = length(rankPP(rankPP<=i))/length(rankPP);
end

figure; plot(cmcTab*100,'.-r');
xlabel('rank');
ylabel('identification rate (%)');

results.modelType = 'PLS';
results.respVec = respVec;
results.DictClass = DictClass;
results.DictData = DictData;
results.expNum = expNum;
results.cmc = cmcTab;
results.kPLS = kParam;
results.rank = rankPP;
results.modelNet = netName;
results.rank1 = length(rankPP(rankPP<=1))/length(rankPP);
results.rank15 = length(rankPP(rankPP<=15))/length(rankPP);
results.rank30 = length(rankPP(rankPP<=30))/length(rankPP);
nameResults = strcat('resultsPLS',num2str(kParam),'.mat');
save(fullfile(paths.results,expNum,nameResults),'results');
fprintf('results, scores and cmc saved');
end
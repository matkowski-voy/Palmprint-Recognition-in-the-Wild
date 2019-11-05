
function []=buildPLSmodels(expNum,netName,kParam)

% expNum = 'exp1';
% netName = 'net-epoch-10.mat';

pathSet = fullfile(pwd,'results',expNum,'features','train');
mkdir(fullfile(pwd,'results',expNum,'pls'));
pathSave = fullfile(pwd,'results',expNum,'pls');


A = importdata(fullfile(pathSet,'set.mat'));
A = A.Vec;

fprintf('dataset loaded\n');

gpuDevice(1);

% kParam = 50;        % parameters PLS

tic
labels = unique(A(:,end-2));
for i=1:length(labels)
%     fprintf('iter: %d\n',i);
indPos = (A(:,end-2) == labels(i));
positive = A(indPos,:); % choose only positive classes form the same hand
negative = A(~indPos,:);

positive = gpuArray(positive);
negative = gpuArray(negative);

X = [positive(:,1:end-3); negative(:,1:end-3)];
Y = [ones(size(positive(:,1:end-3),1),1); ones(size(negative(:,1:end-3),1),1)*-1];

clear positive
clear negative
[X, xMu, xSigma] = zscore(X);
[Y, yMu, ySigma] = zscore(Y);

% tic
b = pls(X,Y,kParam,true);
% toc

save(fullfile(pathSave,strcat('classifier',num2str(labels(i)),'_',netName(1:end-4),'.mat')),...
                'b','xMu','xSigma','yMu','ySigma','kParam','netName')
            strcat('classifier',num2str(labels(i)),'.mat');

end
toc
fprintf('PLS models built and saved\n')
end
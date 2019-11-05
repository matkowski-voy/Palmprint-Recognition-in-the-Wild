% build imdb

function []=augment_build_imdb(expNum,dbName,imgType,dataListName)
fprintf('building imdb\n')
% expNum = 'exp1'; 
% dbName = 'NTU-PI-demo'; 
% imgType = 'img'; 
% dataListName = 'DataNclass.mat';
% define affine transformation parameters for image augmentation
ang = [-90 -20 -5 0 5 20 90];  % angles 
s = [0.8 0.9 1 1.1 1.2]; % scale
tx = [-.2 -.1 -.15 .1 .15 .2]; % translation x
ty = [-.2 -.1 -.15 .1 .15 .2]; % translation y

pathTrain = fullfile('../databases/',dbName,'/flip/',imgType,'/train/');
pathTest = fullfile('../databases/',dbName,'/flip/',imgType,'/test/');
mkdir(fullfile(pwd,'/imdbDataList/',expNum));
pathSave = fullfile(pwd,'/imdbDataList/',expNum,'/');
fileTrain = dir(pathTrain); fileTrain = fileTrain(3:end);
fileTest = dir(pathTest); fileTest = fileTest(3:end);

indTrain = 1;
indTest = 1;
indMetaTransform = 1;
identityTransformation(:,:,1) = 1;
identityTransformation(:,:,2) = 0;
identityTransformation(:,:,3) = 0;
identityTransformation(:,:,4) = 1;
identityTransformation(:,:,5) = 0;
identityTransformation(:,:,6) = 0;

for i=1:length(fileTrain)
    c1 = strfind(fileTrain(i).name,'Class');
    c2 = strfind(fileTrain(i).name,'.'); 
    tabId(i) = str2num(fileTrain(i).name(c1+5:c2-1));
end

labelDict = unique(tabId);

% build imdb training set
for i=1:length(fileTrain)
    c1 = strfind(fileTrain(i).name,'Class');
    c2 = strfind(fileTrain(i).name,'.'); 
    classId = str2num(fileTrain(i).name(c1+5:c2-1));
    
    imdb.tr.image{indTrain} = fileTrain(i).name;
    classId = find(classId == labelDict);
    imdb.tr.label(indTrain) = classId;
    indTrain = indTrain + 1;
end


%% build imdb test set
for i=1:length(fileTest)
    c1 = strfind(fileTest(i).name,'Class');
    c2 = strfind(fileTest(i).name,'.');  
    classId = str2num(fileTest(i).name(c1+5:c2-1));
    classId = find(classId == labelDict);
    imdb.val.image{indTest} = fileTest(i).name;
    imdb.val.label(indTest) = classId;
    indTest = indTest + 1;      
end

z=unique(imdb.val.label);
z2=unique(imdb.tr.label);

%% consistency (closed set) checker
% if an image from test/probe set is not in train/gallery set then return 
% pause and return this image
if(iscell(z))
for i=1:length(z)
    flag = false;
    for j=1:length(z2)
        if(strcmp(z{i},z2{j}))
            flag = true;
        end
    end
    if(flag == false)
        z{i}
        pause
    end
end
else
    for i=1:length(z)
    flag = false;
    for j=1:length(z2)
        if(z(i)==z2(j))
            flag = true;
        end
    end
    if(flag == false)
        z(i)
        pause
    end
    end
end

% input the set of transofmrations to apply to images on the fly
    for t1=1:length(ang)
        theta(:,:,1) = cosd(ang(t1));
        theta(:,:,2) = sind(ang(t1));
        theta(:,:,3) = -sind(ang(t1));
        theta(:,:,4) = cosd(ang(t1));
        theta(:,:,5) = 0;
        theta(:,:,6) = 0;
        imdb.meta.setTransformation(:,:,:,indMetaTransform) ...
            = theta;
        indMetaTransform = indMetaTransform + 1;
    end
    for t2=1:length(s)
        theta(:,:,1) = s(t2)*cosd(0);
        theta(:,:,2) = s(t2)*sind(0);
        theta(:,:,3) = s(t2)*-sind(0);
        theta(:,:,4) = s(t2)*cosd(0);
        theta(:,:,5) = 0;
        theta(:,:,6) = 0;
        imdb.meta.setTransformation(:,:,:,indMetaTransform) ...
            = theta;
        indMetaTransform = indMetaTransform + 1;
    end
    for t3=1:length(tx)
        theta(:,:,1) = cosd(0);
        theta(:,:,2) = sind(0);
        theta(:,:,3) = -sind(0);
        theta(:,:,4) = cosd(0);
        theta(:,:,5) = 0;
        theta(:,:,6) = tx(t3);
        imdb.meta.setTransformation(:,:,:,indMetaTransform) ...
            = theta;
        indMetaTransform = indMetaTransform + 1;
    end
    for t4=1:length(ty)
        theta(:,:,1) = cosd(0);
        theta(:,:,2) = sind(0);
        theta(:,:,3) = -sind(0);
        theta(:,:,4) = cosd(0);
        theta(:,:,5) = ty(t4);
        theta(:,:,6) = 0;
        imdb.meta.setTransformation(:,:,:,indMetaTransform) ...
            = theta;
        indMetaTransform = indMetaTransform + 1;
    end

% prepare imdb struct to save
imdb.meta.labelDict = labelDict;
imdb.meta.classes = unique(z2);
imdb.meta.transformation.ang = ang;
imdb.meta.transformation.s = s;
imdb.meta.transformation.tx = tx;
imdb.meta.transformation.ty = ty;
imdb.meta.imgType = imgType;
imdb.meta.dataSetName = dbName;
imdb.meta.numClassesGallery = length(z2);
imdb.meta.numClassesProbe = length(z);

save(fullfile(pathSave,dataListName),'imdb');
fprintf('imdb saved\n');
z=unique(imdb.val.label);
z2=unique(imdb.tr.label);
fprintf('unique classes in training/gallery set %d:\n',length(z2));
fprintf('unique classes in testing/probe set %d:\n',length(z));
fprintf('affine transformation parameters:\n');
imdb.meta.transformation

end

% transform and save images and save imdb list file
% by voy
% this script builds imdb and transforms images for ROI-LAnet training
% script takes already fliped train and test images and rotate
% script takes masked images and flip them and rotate
% script takes landmarks and for the left hand flips them
% preprocessed images are saved in ../databases/',dbName,'flip/imgGen
% imdb struct is saved in ../ROI-LAnet/imdbDataList

clear all;close all;

dbName = 'NTU-PI-demo'; % NTU-PI-v1
visPoints = false; % set true if want to preview points

pathTrain = fullfile('../databases/',dbName,'flip/img/train');
pathTest = fullfile('../databases/',dbName,'flip/img/test');
pathLandmarks = fullfile('../databases/',dbName,'landmarks');
pathMask = fullfile('../databases/',dbName,'imgMask');
mkdir(fullfile('../databases/',dbName,'flip/imgGen'));
pathGen = fullfile('../databases/',dbName,'flip/imgGen');
pathSaveImdb = fullfile('../ROI-LAnet/imdbDataList');

fileTrain = dir(pathTrain); fileTrain=fileTrain(3:end);
fileTest = dir(pathTest); fileTest=fileTest(3:end);

five_points = false;
nine_points = true;

ind = 1;
rng(2018);
tic
for i = 1:length(fileTrain)  
   c = strfind(fileTrain(i).name,'-');
   c2 = strfind(fileTrain(i).name,'.');
   c3 = strfind(fileTrain(i).name,'L'); % this is how left hands are indicated
   c4 = strfind(fileTrain(i).name,'Class');
   
    imdb.tr.image{ind} = fileTrain(i).name;
    load(fullfile(pathLandmarks,strcat(fileTrain(i).name(1:c4-2),'.mat')));
    I = imread(fullfile(pathTrain,fileTrain(i).name));
    I = imresize(I,[224 224]);

    if ~isempty(c3)
%         fprintf('left hand\n');
        thetaTps(:,:,1:9) = thetaTps(:,:,1:9)*-1;
        thetaTpsTmp = thetaTps;
        thetaTps(:,:,1:3) = thetaTpsTmp(:,:,7:9);
        thetaTps(:,:,7:9) = thetaTpsTmp(:,:,1:3);
        thetaTps(:,:,10:12) = thetaTpsTmp(:,:,16:18);
        thetaTps(:,:,16:18) = thetaTpsTmp(:,:,10:12);
    else
%     fprintf('right hand\n');
    end
    if(five_points == true)
    thetaTps(:,:,[2 5 6 8 11 14 15 17]) = [];
    end
    imdb.tr.label(:,:,:,ind) = thetaTps;
    points = squeeze(thetaTps);
    if(nine_points == true)
    p2do = ones(3,9);
    p2do(1,:) = points(1:9);
    p2do(2,:) = points(10:18);
    end
    if(five_points == true)
    p2do = ones(3,5);
    p2do(1,:) = points(1:5);
    p2do(2,:) = points(6:10);
    end
    ind = ind + 1;
    
    
    Imask = imread(fullfile(pathMask,strcat(fileTrain(i).name(1:c4-2),fileTrain(i).name(c2:end))));
    if ~isempty(c3)
        Imask  = flip(Imask,2);
    end
        
    newNameMask = strcat(fileTrain(i).name(1:c4-1),'_Mask_',fileTrain(i).name(c4:c2-1),fileTrain(i).name(c2:end));
    imwrite(Imask,fullfile(pathGen,newNameMask));
    imwrite(I,fullfile(pathGen,fileTrain(i).name));
    
    imdb.tr.image{ind} = newNameMask;
    imdb.tr.label(:,:,:,ind) = thetaTps;   
    ind = ind + 1;
    
    
    theta(:,:,1,1) = 1;
    for ang=[90 180 270]

        Rmat = [cosd(ang) sind(ang) 0;
            -sind(ang) cosd(ang) 0;
            0 0 1];
      
        theta(:,:,1:2) = Rmat(1,1:2);
        theta(:,:,3:4) = Rmat(2,1:2);
        theta(:,:,5) = Rmat(1,3);
        theta(:,:,6) = Rmat(2,3);
        theta = single(theta);
        p2d = Rmat*p2do;
        
        if(nine_points == true)
        newThetaTps(1:9) = p2d(1,:);
        newThetaTps(10:18) = p2d(2,:);
        end
        if(five_points == true)
        newThetaTps(1:5) = p2d(1,:);
        newThetaTps(6:10) = p2d(2,:);
        end
        
        I2 = imrotate(I,ang);
        
        if(visPoints == true)
        maxA = size(I,2);
        minA = 1;
        pNew = (maxA-minA)/(1 + 1).*(p2d-1) + maxA;
        figure; imshow(I);
        figure; imshow(I2); hold on;
        plot(pNew(1,:),pNew(2,:),'.r');
        pause
        close all
        end
        
        newName = strcat(fileTrain(i).name(1:c4-1),num2str(ang),fileTrain(i).name(c4:c2-1),fileTrain(i).name(c2:end));
        imwrite(I2,fullfile(pathGen,newName));
        imdb.tr.image{ind} = newName;
        imdb.tr.label(:,:,:,ind) = newThetaTps;   
        ind = ind + 1;
        
        angMask = ang + floor(rand*90);
        Rmat = [cosd(angMask) sind(angMask) 0;
            -sind(angMask) cosd(angMask) 0;
            0 0 1];
      
        p2d = Rmat*p2do;
        if(nine_points == true)
        newThetaTps(1:9) = p2d(1,:);
        newThetaTps(10:18) = p2d(2,:);
        end
        if(five_points == true)
        newThetaTps(1:5) = p2d(1,:);
        newThetaTps(6:10) = p2d(2,:);
        end
               
        I2_Mask = imrotate(Imask,angMask,'crop');

        newName = strcat(fileTrain(i).name(1:c4-1),'_Mask_',num2str(angMask),fileTrain(i).name(c4:c2-1),fileTrain(i).name(c2:end));
        imwrite(I2_Mask,fullfile(pathGen,newName));
        imdb.tr.image{ind} = newName;
        imdb.tr.label(:,:,:,ind) = newThetaTps;   
        ind = ind + 1;
        
    end
    
    
   
end

ind2 = 1;
for i = 1:length(fileTest)
    
   c = strfind(fileTest(i).name,'-');
   c2 = strfind(fileTest(i).name,'.');
   c3 = strfind(fileTest(i).name,'L');
   c4 = strfind(fileTest(i).name,'Class');
   
    imdb.val.image{ind2} = fileTest(i).name;
    load(fullfile(pathLandmarks,strcat(fileTest(i).name(1:c4-2),'.mat')));
    
    I = imread(fullfile(pathTest,fileTest(i).name));

    if ~isempty(c3)
        thetaTps(:,:,1:9) = thetaTps(:,:,1:9)*-1;
        thetaTpsTmp = thetaTps;
        thetaTps(:,:,1:3) = thetaTpsTmp(:,:,7:9);
        thetaTps(:,:,7:9) = thetaTpsTmp(:,:,1:3);
        thetaTps(:,:,10:12) = thetaTpsTmp(:,:,16:18);
        thetaTps(:,:,16:18) = thetaTpsTmp(:,:,10:12);
    end
    if(five_points == true)
    thetaTps(:,:,[2 5 6 8 11 14 15 17]) = [];
    end
    imdb.val.label(:,:,:,ind2) = thetaTps;
    ind2 = ind2 + 1;
    
    imwrite(I,fullfile(pathGen,fileTest(i).name));

end

save(fullfile(pathSaveImdb,'DataAlign.mat'),'imdb');
fprintf('augmented images saved\n');
fprintf('imdb saved\n');
toc

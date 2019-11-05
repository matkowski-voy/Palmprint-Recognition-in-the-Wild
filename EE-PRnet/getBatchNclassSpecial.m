function inputs = getBatchNclassSpecial(imdb,batchIdx,trSetIdx,at_epoch_on)

imsize = [56 56];
imsize224 = [224 224];

avRGB = [123.68 116.78 103.94];

label_batch = single(imdb.label(1,batchIdx));
numSetTransforms = size(imdb.setTransformation,4);
tf_idx = randi(numSetTransforms,1,length(batchIdx));
transformation_batch = gpuArray(single(imdb.setTransformation(:,:,:,tf_idx)));
if(batchIdx(1) < trSetIdx+1)
img = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize,'SubtractAverage',avRGB,'Pack','Saturation',.9,'Contrast',0.7);
img2ROI_batch = img{1};
img = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize224,'SubtractAverage',avRGB,'Pack','Saturation',.9,'Contrast',0.7);
img2Align_batch = img{1}; 

img2ROI_batch = gpuArray(img2ROI_batch);
img2Align_batch = gpuArray(img2Align_batch);
label_batch =  gpuArray(label_batch);

    if(imdb.epoch_cnt > at_epoch_on) 
        G = dagnn.AffineGridGenerator('Ho',imsize(1),'Wo',imsize(2));
        Sampler = dagnn.BilinearSampler;
        T_G = G.forward({transformation_batch});
        img2ROI_batch = Sampler.forward({img2ROI_batch,T_G{1}});
        img2ROI_batch = img2ROI_batch{1};
        % 
        G = dagnn.AffineGridGenerator('Ho',imsize224(1),'Wo',imsize224(2));
        Sampler = dagnn.BilinearSampler;
        T_G = G.forward({transformation_batch});
        img2Align_batch = Sampler.forward({img2Align_batch,T_G{1}});
        img2Align_batch = img2Align_batch{1};
    end

else
img = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize,'SubtractAverage',avRGB,'Pack');
img2ROI_batch = img{1};
img = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize224,'SubtractAverage',avRGB,'Pack');
img2Align_batch = img{1}; 
img2ROI_batch = gpuArray(img2ROI_batch);
img2Align_batch = gpuArray(img2Align_batch);
label_batch =  gpuArray(label_batch);
end

% img2ROI_batch = gpuArray(img2ROI_batch);
% img2Align_batch = gpuArray(img2Align_batch);
% label_batch =  gpuArray(label_batch);


inputs = {'label', label_batch, 'ANInput', img2ROI_batch, 'Input224', img2Align_batch};



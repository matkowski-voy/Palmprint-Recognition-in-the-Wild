function inputs = getBatchNclass(imdb,batchIdx,trSetIdx)

imsize = [112 112];
avimnet =[123.68 116.78 103.94];
label_batch = single(imdb.label(1,batchIdx));

if(batchIdx(1) < trSetIdx)
    im2ROI = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize,'SubtractAverage',avimnet,'Saturation',.75,'Contrast',.75,'Pack');
    im2ROI_batch = im2ROI{1};
else
    im2ROI = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize,'SubtractAverage',avimnet,'Pack');
    im2ROI_batch = im2ROI{1};
end

%if ~isempty(topts.gpus)
im2ROI_batch = gpuArray(im2ROI_batch);
label_batch =  gpuArray(label_batch);  
%end

inputs = {'label', label_batch, 'Input', im2ROI_batch};
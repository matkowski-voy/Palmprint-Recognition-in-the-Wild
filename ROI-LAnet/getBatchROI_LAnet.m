function inputs = getBatchROI_LAnet(imdb,batchIdx)

imsize = [56 56];
labelBatch = single(imdb.label(:,:,:,batchIdx));
img = vl_imreadjpeg(imdb.image(batchIdx),'resize',imsize,'SubtractAverage',[123.68 116.78 103.94],'Pack','Contrast',0.5,'Saturation',0.5);

imgBatch = gpuArray(img{1});
labelBatch =  gpuArray(labelBatch) ;

inputs = {'thetaGt', labelBatch, 'ANInput', imgBatch};
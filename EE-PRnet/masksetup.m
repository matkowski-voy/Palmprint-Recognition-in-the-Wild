addpath('../matlab/');
addpath('../matlab/matconvnet');
addpath('../matlab/matconvnet/matlab');
addpath('../preTrained_VGG-16');
addpath('../functions');
addpath('../preTrainedNetworks');
% you may need to compile and setup these too
% vl_compilenn('enableGPU',true,'cudaMethod','nvcc')
run vl_setupnn; 

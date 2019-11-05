function [imgStruct] = idxImages(path2evidence,path2gallery)

file = dir(path2gallery); file=file(3:end);
file2 = dir(path2evidence); file2=file2(3:end);
for i=1:length(file)   
    c1 = strfind(file(i).name,'Class');
    c2 = strfind(file(i).name,'.');
    classId(i) = str2num(file(i).name(c1+5:c2(1)-1));
    image_name{i} = file(i).name;
end

imgStruct.path2evidence = path2evidence;
imgStruct.path2gallery = path2gallery;
imgStruct.classId = classId;
imgStruct.image_name = image_name;
imgStruct.input_name = file2(1).name;

end

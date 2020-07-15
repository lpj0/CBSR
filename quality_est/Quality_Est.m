clear;
addpath(genpath('./.'));
 std_luminance_quant_tbl = ...
[16,  11,  10,  16,  24,  40,  51,  61;
12,  12,  14,  19,  26,  58,  60,  55;
14,  13,  16,  24,  40,  57,  69,  56;
14,  17,  22,  29,  51,  87,  80,  62;
18,  22,  37,  56,  68, 109, 103,  77;
24,  35,  55,  64,  81, 104, 113,  92;
49,  64,  78,  87, 103, 121, 120, 101;
72,  92,  95,  98, 112, 100, 103,  99];

folderCur = '.\';

ext        =  {'*.JPEG','*.jpeg','*.jpg','*.png','*.bmp'};
imgPaths  =  [];
for i = 1 : length(ext)
    imgPaths = cat(1,imgPaths,dir(fullfile(folderCur, ext{i})));
end

idx = 0;
for im_idx = 1:numel(imgPaths)
    im_path = fullfile(folderCur, imgPaths(im_idx).name);
%     im_path_new = fullfile('tmp/', [imgPaths(im_idx).name(1:end-4) '.jpg']);
    
%     PC = randi(100);
%     im = imread(im_path);
%     imwrite(im, im_path_new, 'jpg', 'quality', PC);
%     try
        jpeg_decompress_struct_cinfo = jpeg_read(im_path);
        idx = idx+1;
%     catch
%         continue;
%     end
    
%     delete(im_path_new);
    
    quant_tables = jpeg_decompress_struct_cinfo.quant_tables{1};
    linear_quality = (quant_tables*100-50)./std_luminance_quant_tbl;
    linear_quality_a = ceil(5000./linear_quality(linear_quality>100));
    linear_quality_b = 100 -  ceil(linear_quality( linear_quality<100 & linear_quality>1)/2);
    linear_quality_c = linear_quality(linear_quality==100);
    linear_quality_d = linear_quality(linear_quality==1);
    quality = [linear_quality_a; linear_quality_b; linear_quality_c; linear_quality_d];
    q_min = min(quality(:));
    table=tabulate(quality);
    [F,I]=max(table(:,2));
    result=table(I,1);
    I=find(table(:,2)==F);
%     linear_quality(linear_quality>100) = ceil(5000./linear_quality(linear_quality>100));
    
%     quality = 100 - mean(ceil(linear_quality(:)/2));
    if I < 50
       I = I-1;
    end
    fprintf('%d %d!\n', I, q_min);
    PC_pre(im_idx) = mean(I);
    PC_pre_min(im_idx) = q_min;
end
disp([mean(PC_pre) mean(PC_pre_min)]);




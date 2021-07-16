%% find images
predicted_image_path = '\\research.wpi.edu\leelab\Junbong\Final_codes\Step2_vUnet\average_hist\predict_wholeframe\all-patients\all-patients\'
predicted_images = dir(predicted_image_path);
predicted_images_cell = struct2cell(predicted_images);
predicted_images_cell = predicted_images_cell(1, :);
predicted_images_cell_only_png = contains(predicted_images_cell,'.png');
predicted_images_cell = predicted_images_cell(predicted_images_cell_only_png);

f1_cell_array = {length(predicted_images_cell), 4};

%% image loading
for image_counter=1:length(predicted_images_cell)
cur_image=predicted_images_cell{image_counter};
I = imread(strcat(predicted_image_path, cur_image));
%figure, imshow(I), title('original image');

%% image processing
filled_I = imfill(I, 'holes');
%figure, imshow(filled_I), title('fill hole');

I_threshold = filled_I >= 255;
%figure, imshow(I_threshold), title('threshold image');

seD = strel('diamond',2);
BWfinal = imerode(I_threshold,seD);
BWfinal = imerode(BWfinal,seD);
BWfinal = imerode(BWfinal,seD);
BWfinal = imerode(BWfinal,seD);
BWfinal = imerode(BWfinal,seD);
BWfinal = imerode(BWfinal,seD);
BWfinal = imerode(BWfinal,seD);
%figure, imshow(BWfinal), title('eroded image');
BWfinal = imdilate(BWfinal,seD);
BWfinal = imdilate(BWfinal,seD);
BWfinal = imdilate(BWfinal,seD);
BWfinal = imdilate(BWfinal,seD);
BWfinal = imdilate(BWfinal,seD);
BWfinal = imopen(BWfinal,seD);
%figure, imshow(BWfinal), title('dillated image');

BWfinal = imfill(BWfinal, 'holes');
%figure, imshow(BWfinal), title('BWdfill');

[B,L] = bwboundaries(BWfinal, 'noholes');
%figure, imshow(label2rgb(L, @jet, [.5 .5 .5]));

%% get edges
%BWoutline=edge(BWfinal, 'canny');
%BWoutline = bwperim(BWfinal);
BWoutline = boundarymask(BWfinal);
Segout = I;
Segout(BWoutline) = 255; 

%figure, imshow(Segout), title('outlined original image');

%% get follicular and macrophage cells
labeled_image_name1 = strrep(cur_image,'predict','');
labeled_image_name = strrep(labeled_image_name1,'.png','-labeled.png');
labeled_image_path = '\\research.wpi.edu\leelab\Junbong\DataSet_label\macrophage_follicular\mask\';
output_path = '\\research.wpi.edu\leelab\Junbong\DataSet_label\macrophage_follicular\output\';
labeled_I = imread(strcat(labeled_image_path, labeled_image_name));

[x1,y1] = size(I);
resized_labeled_I = imresize(labeled_I, [x1 y1]);
%figure, imshow(labeled_I);

redchannel = resized_labeled_I(:, :, 1);
redchannel_logical = redchannel == 255;
redchannel_fused = imfuse(redchannel_logical,BWoutline);
%figure, imshow(redchannel_fused), title('redchannel_fused');
imwrite(redchannel_fused, strcat(output_path, 'redchannel_fused', int2str(image_counter), '.png'));

greenchannel = resized_labeled_I(:, :, 2);
greenchannel_logical = greenchannel == 255;
greenchannel_fused = imfuse(greenchannel_logical,BWoutline);
%figure, imshow(greenchannel_fused), title('greenchannel_fused');
imwrite(greenchannel_fused, strcat(output_path, 'greenchannel_fused', int2str(image_counter), '.png'));

%%
overlappedRed = redchannel_logical & BWfinal;
overlappedGreen = greenchannel_logical & BWfinal;
%figure, imshow(overlappedRed), title('overlappedRed image');
%figure, imshow(overlappedGreen), title('overlappedGreen image');

len_follicular = length(regionprops(redchannel_logical))
len_macrophage = length(regionprops(greenchannel_logical))
total_cells = len_follicular + len_macrophage

len_overlapped_follicular = length(regionprops(overlappedRed))
len_overlapped_macrophage = length(regionprops(overlappedGreen))

% statistics
tp = len_overlapped_follicular
fp = len_overlapped_macrophage;
fn = len_follicular - len_overlapped_follicular;
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(precision*recall)/(precision+recall)
f1_cell_array{image_counter, 1} = labeled_image_name1; 
f1_cell_array{image_counter, 2} = precision;
f1_cell_array{image_counter, 3} = recall;
f1_cell_array{image_counter, 4} = f1_score;

end
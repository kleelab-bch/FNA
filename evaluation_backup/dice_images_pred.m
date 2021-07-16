% Author Junbong Jang
% Date 10/8/2019
clear all;
%dataset = {'gp2781-first', 'gp2781-fourth', 'gp2781-third', 'ha2779-first', 'ha2779-third'};
dataset = {'all-patients'};
dice_coefficient_cell = cell(20,2);
for n = 1 : length(dataset)
    folder = dataset(n)
	mask_path = strcat('../assets/', folder, '/mask/');
	predict_path = strcat('../vUnet/average_hist/predict_wholeframe/', folder, '/', folder, '/');
    generate_path = strcat('../vUnet/average_hist/predict_wholeframe/', folder, '/');

	mask_path = mask_path{1};
	predict_path = predict_path{1};
    generate_path = generate_path{1};
    filesStructure = dir(predict_path);
	allFileNames = {filesStructure(:).name};
    
    threshold_vals = [0,25,50,75,100,125,150,175,200];
    for threshold_index = 1:length(threshold_vals)
        threshold_val = threshold_vals(threshold_index);
        generate_path = [generate_path, 'threshold_', num2str(threshold_val), '/']
        mkdir(generate_path);

        similarity_total = 0;
        counter = 1;

        for k = 1 : length(allFileNames)
            fileName = allFileNames{k}
            imageFileBool = contains(fileName,'.png');
            if imageFileBool
                % get original iamge
                filePath = strcat(mask_path, strrep(fileName,'predict',''))
                I = imread(filePath);
                I = rgb2gray(I);
                I_binary = (I==76);
    % 			I_binary = imbinarize(I);
                [row, col] = size(I_binary);
                I_binary = I_binary(31:row, 31:col);

                % get predicted image
                predFilePath = strcat(predict_path, fileName);
                I2 = imread(predFilePath);
                %I2_binary = imbinarize(I2);
                I2_binary = I2 > threshold_val;
                I2_binary = I2_binary(31:row, 31:col);
                %% image processing
                I2_final = bwareaopen(I2_binary, 2000);

                %% calculate similarity
                similarity = dice(I_binary, I2_final)
                similarity_total = similarity_total + similarity;
                figure(1), clf
                hAx = axes;
                C = imfuse(I_binary, I2_final); % where gcf is created
                imshow(C, 'Parent', hAx, 'Border','Loose');
                title(hAx, ['Dice Index = ' num2str(similarity)], 'FontSize', 14, 'Color','b');
                imlegend([1 0 1; 0 1 0; 1 1 1], {'Predict'; 'True'; 'Overlap'});
                saveas(gcf, strcat(generate_path, '/compare-',fileName));
                imwrite(I2_final, [generate_path, fileName]);

                dice_coefficient_cell{counter, 1} = fileName;
                dice_coefficient_cell{counter, 2} = num2str(similarity);
                counter = counter + 1
            end
        end
        similarity_mean = similarity_total / counter
    end
end

function imlegend(colorArr, labelsArr)
    % For instance if two legend entries are needed:
    % colorArr =
    %   Nx3 array of doubles. Each entry should be an RGB percentage value between 0 and 1
    %
    % labelsArr =
    %   1×N cell array
    %     {'First name here'}    {'Second name here'}    {'etc'}
    hold on;
    for ii = 1:length(labelsArr)
      % Make a new legend entry for each label. 'color' contains a 0->255 RGB triplet
      scatter([],[],1, colorArr(ii,:), 'filled', 'DisplayName', labelsArr{ii});
    end
    hold off;
    lgnd = legend();  
    set(lgnd,'color',[127 127 127]/255);
end
function data = preprocessData(data,targetSize)
for num = 1:size(data,1)
    I = data{num,1};
    imgSize = size(I);
    bboxes = data{num,2};
    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    data(num,1:2) = {I,bboxes};
end
end

% PREPARE TRAINING DATA
imageHeight = 640;
imageWidth = 640;
classes = {'Amortecedor', 'Isolador', 'Torre de transmissao'};

% Load a .mat file containing information about the vehicle dataset to use for training. The information stored in the .mat file is a table. The first column contains the training images and the remaining columns contain the labeled bounding boxes.
data = load("C:\Users\gusta\Downloads\rede_copel.mat");
trainingData = data.rede_copel;
% Create an imageDatastore using the files from the table.
strData = cellstr(trainingData.path);
imds = imageDatastore(strData);

bboxes = trainingData.bbox;
labels = trainingData.class;

% Convert normalized bboxes to pixel values
bboxesPixel = zeros(size(bboxes)); % Initialize the pixel bounding boxes array
for i = 1:size(bboxes, 1)
    bboxesPixel(i, 1) = round(bboxes(i, 1) * imageWidth);  % x0 in pixels
    bboxesPixel(i, 2) = round(bboxes(i, 2) * imageHeight); % y0 in pixels
    bboxesPixel(i, 3) = round(bboxes(i, 3) * imageWidth);  % width in pixels
    bboxesPixel(i, 4) = round(bboxes(i, 4) * imageHeight); % height in pixels
end

% Convert bboxes to cell array of cells containing [x0 y0 w h]
bboxesCell = cell(size(bboxes, 1), 1); % Initialize cell array
for i = 1:size(bboxes, 1)
    bboxesCell{i} = bboxesPixel(i, :); % Assign each row to a cell
end

% Alternatively, use string vectors
labels = labels + 1;  % This assumes labels are 0-based, adjust if necessary
for i = 1:size(labels, 1)
    if labels(i) > 0 && labels(i) <= numel(classes)
        labelCell{i} = string(classes(labels(i)));  % Map label to class name
    else
        error('Invalid label index at row %d: %d', i, labels(i));
    end
end

% Create a table
data.rede_copel = table(strData, bboxesCell, labelCell, 'VariableNames', {'Path', 'BBox', 'Label'});
trainingData = data.rede_copel;

% Create a boxLabelDatastore using the label columns from the table.
blds = boxLabelDatastore(trainingData(:,2:end));

% Combine the datastores.
ds = combine(imds,blds);
read(ds);

% Specify the input size to use for resizing the training images. The size of the training images must be a multiple of 32 for when you use the tiny-yolov4-coco and csp-darknet53-coco pretrained YOLO v4 deep learning networks. You must also resize the bounding boxes based on the specified input size.
inputSize = [224 224 3];

% Resize and rescale the training images and the bounding boxes by using the preprocessData helper function. Also, convert the preprocessed data to a datastore object by using the transform function.
trainingDataForEstimation = transform(ds,@(data)preprocessData(data,inputSize));

% ESTIMATE ANCHOR BOXES
% Estimate the anchor boxes from the training data. You must assign the same number of anchor boxes to each output layer in the YOLO v4 network.
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
area = anchors(:,1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};

% CONFIGURE AND TRAIN YOLOV4    
% Specify the class names and configure the pretrained YOLO v4 deep learning network to be retrained for the new dataset by using yolov4ObjectDetector function.
    detector = yolov4ObjectDetector("tiny-yolov4-coco",classes,anchorBoxes,InputSize=inputSize);
    % Specify the training options and retrain the pretrained YOLO v4 network on the new dataset by using the trainYOLOv4ObjectDetector function.
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.001, ...
    MiniBatchSize=16,...
    MaxEpochs=40, ...
    ResetInputNormalization=false,...
    VerboseFrequency=30);
trainedDetector = trainYOLOv4ObjectDetector(ds,detector,options);

% DETECT WITH A TRAINED MODEL
% Read a test image.
I = imread("img.png");
% Use the fine-tuned YOLO v4 object detector to detect vehicles in a test image and display the detection results.
[bboxes, scores, labels] = detect(trainedDetector,I,Threshold=0.05);
detectedImg = insertObjectAnnotation(I,"Rectangle",bboxes,labels);
figure
imshow(detectedImg)

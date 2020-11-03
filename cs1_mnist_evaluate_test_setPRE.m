%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace

% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

predictions = zeros(200,1);
outliers = zeros(200,1);

% loop through the test set, figure out the predicted number
sub(1:784) = 200;
for i = 1:size(test)
    if test(i,1)~=0
        outliers(i,1)=1;
        test(i,1:784) = test(i,1:784) - sub;
        test(i,1) = 1;
    end
end

for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
sub(1:784) = 200;
for i = 1:size(test)
    if test(i,1)~=0
        outliers(i,1)=1;
        test(i,1:784) = test(i,1:784) + sub;
        test(i,1) = test(i,1) - 1;
    end
end


%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers);
title('Outlier Plot');
xlabel('Picture Index');
ylabel('Outlier Flags');

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

%% The following line provides the number of instances where an entry in correctlabel is
% equal to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

%function [index, vec_distance] = assign_vector_to_centroid(data,centroids)

%end

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
distance = zeros(size(centroids));
for i = 1:size(centroids)
    distance(i) = norm(data - centroids(i,:));
end
[change, ind] = sort(distance);
index = ind(1);
vec_distance = change(1);
end

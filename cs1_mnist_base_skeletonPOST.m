
clear all;
close all;

%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.


%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200_woutliers.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array train consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k= 10; % set k
max_iter= 30; % set the number of iterations of the algorithm

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(train,k,trainsetlabels);


%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm
assignment = zeros(size(train))';
vec_distance = zeros(size(train))';
centroid_labels = [0,2,3,5,6,8,9,1,2,3,4,5,6,7,8,9,1,2,3,6,8,0,1,2,3,5,6,8,3,0,2,3,4,5,6,7,8,9,0,2,3,4,5,6,8,9,0,2,3,4,5,6,8,2,3,5,7,9,0,1,2,3,4,5,7,8,9,1,2,3,4,5,7,8,9];

for iter=1:max_iter
    for j = 1:size(train)
        [assignment(j), vec_distance(j)] = assign_vector_to_centroid(train(j,:),centroids);
        cost_iteration(iter)=cost_iteration(iter) + (vec_distance(j))^2;
    end
   
    for l = 1:size(train)
        train(l,785) = assignment (l);
    end
    
    centroids = update_Centroids(train,k);
    
end

for p = 1:size(train)
    train(p,1) = train(p,1) + p;
end


one = 1;
two = 1;
three = 1;
four = 1;
five = 1;
six = 1;
seven = 1;
eight = 1;
nine = 1;
ten = 1;
for i = 1:size(train)
    if train(i,785) == 1
        cent1(one,:) = train(i,:);
        one = one + 1;
    end
    if train(i,785) == 2
        cent2(two,:) = train(i,:);
        two = two + 1;
    end
    if train(i,785) == 3
        cent3(three,:) = train(i,:);
        three = three + 1;
    end
    if train(i,785) == 4
        cent4(four,:) = train(i,:);
        four = four + 1;
    end
    if train(i,785) == 5
        cent5(five,:) = train(i,:);
        five = five + 1;
    end
    if train(i,785) == 6
        cent6(six,:) = train(i,:);
        six = six + 1;
    end
    if train(i,785) == 7
        cent7(seven,:) = train(i,:);
        seven = seven + 1;
    end
    if train(i,785) == 8
        cent8(eight,:) = train(i,:);
        eight = eight + 1;
    end
    if train(i,785) == 9
        cent9(nine,:) = train(i,:);
        nine = nine + 1;
    end
    if train(i,785) == 10
        cent10(ten,:) = train(i,:);
        ten = ten + 1;
    end
end


cent1new = average_centroids(cent1,trainsetlabels);
cent2new = average_centroids(cent2,trainsetlabels);
cent3new = average_centroids(cent3,trainsetlabels);
cent4new = average_centroids(cent4,trainsetlabels);
cent5new = average_centroids(cent5,trainsetlabels);
cent6new = average_centroids(cent6,trainsetlabels);
cent7new = average_centroids(cent7,trainsetlabels);
cent8new = average_centroids(cent8,trainsetlabels);
cent9new = average_centroids(cent9,trainsetlabels);
cent10new = average_centroids(cent10,trainsetlabels);

centroids = [cent1new',cent2new',cent3new',cent4new',cent5new',cent6new',cent7new',cent8new',cent9new',cent10new']';


            

%% This section of code plots the k-means cost as a function of the number
% of iterations
for i = 1:max_iter
    nums(i)=i;
end
figure;
plot(nums,cost_iteration,'-o');
title('K-means Costs');
xlabel('Iteration');
ylabel('Cost');

%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(20));

for ind=1:20
    
    centr=centroids(ind,[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title(strcat('Centroid ',num2str(ind)))
    axis image;
end

save('classifierdata.mat','centroid_labels','centroids');

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% *Feel free to experiment.*
% Note that this function takes two inputs and emits one output (y).

function y=initialize_centroids(data,num_centroids,correctans)

avg(num_centroids,1:785)=0;
 
for j = 0:9
    add(1,1:785)=0;
    
    pop=0;
    for i = 1:size(data)
        if correctans(i)==j
            add = add+data(i,:);
            pop = pop+1;
        end
    end
    avg(j+1,:)=add./pop;
end
 
 
 
y=avg;
end
%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
distance = zeros(size(centroids));
for i = 1:size(centroids)
    distance(i) = norm(data - centroids(i,:));
end
[change, ind] = sort(distance);
index = ind(1);
vec_distance = change(1);
end


%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.

function new_centroids = update_Centroids(data,K)

new_centroids(1:K,1:785) = 0;

for i = 1:K
    t = 0;
    sumed(1:785) = 0;
    for j = 1:size(data)
        if data(j,785) == i
            sumed = data(j,:) + sumed;
            t = t + 1;
        end
        new_centroids(i,:) = sumed./t;
    end
end
 

end

function ncentroids = average_centroids(data,ans)

num = 0;
for q = 0:9
    sum(1:785) = 0;
    tot = 0;
    for w = 1:size(data)
        if ans(data(w,1),1) == q
            sum = sum+data(w,:);
            tot = tot+1;
        end
    end
    if tot >= 1
        num = num + 1;
        ncentroids(num,:)=sum./tot;
    end
    
end
end
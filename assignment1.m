% Matrix Decompositions in Data Analysis
% Spring semester 2021
%
% Assignment 1
% Due 2 February 2021 at 23:59

% Student's information
% Name:
% Email:
% Student ID #:

%
% This file contains stubs of code to help you to do your 
% assignment. You can fill your parts at the indicated positions
% and return this file as a part of your solution. Remember that
% if you define any functions etc. in other files, you must include 
% those files, as well. You do not have to include standard libraries
% or any files that we provide.
%
% Remember to fill your name and matriculation number above.
%

%% Preamble
% Matlab cannot mix functions and scripts, so the utility functions are
% provided in separate file(s). Similarly, if you want to write your own
% functions (e.g. for Task 2), you must put them on different files.
%
% You can run this script from Matlab with
% > assignment1
% but it is missing 'normalized_data' from Task 3. 


%% TASK 1
data = load('worldclim.txt');
coord = load('coordinates.txt'); % First column is longitude, second latitude

% Familiarize yourself with the data. You can find the attribute (column)
% names of worldclim.txt from worldclim_attributes.txt. Do not include this
% part in your report.

% Compute the SVD
[U, S, V] = svd(data, 'econ');

% Prepare for plotting; the extremes
lonLim = [min(coord(:,1)), max(coord(:,1))];
latLim = [min(coord(:,2)), max(coord(:,2))];

% Plot the first column of U
figure;
geoscatter(coord(:,2), coord(:,1), 1, U(:,1));
geolimits(latLim, lonLim);
colorbar;

% Try different colors
figure;
colormap('hsv');
geoscatter(coord(:,2), coord(:,1), 1, U(:,1));
geolimits(latLim, lonLim);
colorbar;

% Plot the second column
figure;
colormap('hsv');
geoscatter(coord(:,2), coord(:,1), 1, U(:,2));
geolimits(latLim, lonLim);
colorbar;

% YOUR PART STARTS HERE

% YOUR PART ENDS HERE

%% Task 2

% You have to implement the rank selection techniques yourself.
% Remember to use the normalized data!

% If S is the matrix containing the singular values, you can plot them by
plot(diag(S), 'o'); 
% or
plot(diag(S));

% YOUR PART STARTS HERE

% YOUR PART ENDS HERE

%% Task 3
normalized_data = zscore(data);
% We again use the normalized data. If that is contained in matrix D
% we can compute the k-means to 5 clusters with 10 re-starts as
clusters = kmeans(normalized_data, 5, 'Replicates', 10);

% Plotting like in Task 1, but let's use different colormap with five
% colors
figure;
geoscatter(coord(:,2), coord(:,1), 1, clusters);
colormap(lines(5));

% To compare two different clusterings, you should match their cluster
% numbers so that the colors of the plots match. There's a file called
% matching.m that contains a functions that computes the maximum matching
% of the cluster labels. You can use it as follows: (the example is
% commented out as it's not part of your task and you don't need to include
% it to your report)
% clust1 = kmeans(normalized_data, 5);
% clust2 = kmeans(normalized_data, 5); % this is just an example
% figure
% geoscatter(coord(:,2), coord(:,1), 1, clust1);
% title('First clustering');
% % colors of this clustering might not match with the above
% figure
% geoscatter(coord(:,2), coord(:,1), 1, clust2);
% title('Second clustering, original colors');
% % here we use matching to match the colors as well as possible
% figure
% geoscatter(coord(:,2), coord(:,1), 1, matching(clust1, clust2));
% title('Second clustering, matched colors');

% YOUR PART STARTS HERE

% YOUR PART ENDS HERE

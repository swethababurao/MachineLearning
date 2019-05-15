function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
error_min = 0;
r = 0;
c = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
c_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
error_val = zeros(length(sigma_vec),length(c_vec));
for i = 1:length(sigma_vec)
    for j = 1:length(c_vec)
        sigma = sigma_vec(i);
        C = c_vec(j);
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error_val(i,j) = mean(double(predictions ~= yval))
    end
end
error_min = min(error_val(:));
[r,c] = find(error_val == error_min);


sigma = sigma_vec(r);
C = c_vec(c);


% =========================================================================

end

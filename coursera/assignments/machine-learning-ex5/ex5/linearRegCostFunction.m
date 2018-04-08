function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Hypothesis func with the training data and the parameters
H = (X*theta);

% Unregularized Cost func
J = (sum((H - y).^ 2)) / (2*m);

% Regularized cost func
REG = (lambda / (2*m)) * sum(theta(2:size(theta),1).^2);
J = J + REG;

%theta(2:size(theta)) = theta(2:size(theta)) - (lambda .* delta);

E = (H - y);
delta = (X(:, 2:size(theta))' * E)/m;
theta(1) = ((1/m) * (E' * X(:,1)));
theta(2:size(theta)) = delta + theta(2:size(theta)) .* (lambda/m);


% =========================================================================

grad = theta(:);

end

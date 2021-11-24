function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters, sigma)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
k = 2/sum(sigma.^2);
J_history = zeros(num_iters, 1);
X = [ones(m,1) X];

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
hypothesis = X*theta;
dif = (y - hypothesis)./(sigma.^2);
delta = (-2*dif'*X)';
theta = theta - alpha*delta;




    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);
    

end

end

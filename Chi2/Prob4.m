   %Este código realiza Gradient descent para aprende parámetros theta

X = [6;9;15];
y = [3;9;11];
sigma = [1;2;3];
sigma2 = [10;2;3];

errorbar(X, y,sigma, 'bo')
ylabel('y')
xlabel('x')
hold on


theta = zeros(2,1);
iterations = 1000;
alpha = 0.001;

theta = gradientDescent(X, y, theta, alpha, iterations, sigma);

theta2 = gradientDescent(X, y, [0;0], alpha, iterations, sigma2);

X = [ones(3,1) X];

plot(X(:,2), X*theta, 'r-', 'LineWidth', 3)
plot(X(:,2), X*theta2, 'k-', 'LineWidth', 3)
legend('Datos', 'Ajuste Lineal 1', 'Ajuste Lineal 2')

axis([4 16 0 20])
hold off




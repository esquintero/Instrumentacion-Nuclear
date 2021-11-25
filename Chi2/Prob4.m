
%Este código realiza Gradient descent para aprende parámetros theta

X = [6;9;15];
y = [3;9;11];
sigma = [1;2;3];
sigma2 = [10;2;3];

subplot(1,2,1);
errorbar(X, y,sigma, 'bo')
hold on
ylabel('y')
xlabel('x')

theta = zeros(2,1);
iterations = 100;
alpha = 0.0003;

[theta J]= gradientDescent(X, y, theta, alpha, iterations, sigma);

[theta2 Js]= gradientDescent(X, y, [0;0], alpha, iterations, sigma2);

X = [ones(3,1) X];

plot(X(:,2), X*theta, 'r-', 'LineWidth', 3)
plot(X(:,2), X*theta2, 'k-', 'LineWidth', 3)
legend('Datos', 'Ajuste Lineal 1', 'Ajuste Lineal 2')

axis([4 16 0 20])
hold off

fprintf('Parametros theta0 y theta1 calculados por Gradient Descent para el primer ajuste:\n%f,\n%f',theta(1),theta(2));

fprintf('El valor de chi2 para el primer ajuste:\n%f',J(iterations));

fprintf('Parametros theta0 y theta1 calculados por Gradient Descent para el segundo ajuste:\n%f,\n%f',theta2(1),theta2(2));

fprintf('El valor de chi2 para el segundo ajuste:\n%f',Js(iterations));

%%Estimación de mejor parámetro alpha i.e. el que mejor reduce Chi2

subplot(1,2,2);
[theta1 J1]= gradientDescent(X(:,2), y, [0;0], 0.01, iterations, sigma);
plot(1:iterations, J1(1:iterations), 'b');
hold on
title('Comportamiento de Chi2 con diferente alpha')
[theta1 J2]= gradientDescent(X(:,2), y, [0;0], 0.003, iterations, sigma);
plot(1:iterations, J2(1:iterations), 'r');
[theta1 J3]= gradientDescent(X(:,2), y, [0;0], 0.001, iterations, sigma);
plot(1:iterations, J3(1:iterations), 'k');
[theta1 J4]= gradientDescent(X(:,2), y, [0;0], 0.0003, iterations, sigma);
plot(1:iterations, J4(1:iterations), 'g');
[theta1 J5]= gradientDescent(X(:,2), y, [0;0], 0.0001, iterations, sigma);
plot(1:iterations, J5(1:iterations), 'c');
[theta1 J6]= gradientDescent(X(:,2), y, [0;0], 0.00003, iterations, sigma);
plot(1:iterations, J6(1:iterations), 'y');
legend('alpha = 0.01','alpha = 0.003','alpha = 0.001','alpha = 0.0003','alpha = 0.0001','alpha = 0.00003')
hold off

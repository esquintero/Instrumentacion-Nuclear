%Carbono-14
semivida = 55.68; %siglos
Gamma = log(2)/semivida;
tau = semivida/log(2); %vida media
N0 = 100;
dt = 1;

tf = 3*tau;
num_intervalos = round(tf/dt);
timegrid = (1:num_intervalos) * dt;


[inicial intermedio] = ProbPoisson(Gamma,N0,dt) %Cálculo de la prob de poisson para 1 a N(t=0) núcleos

Finicial = DistAcum(N0,inicial);
Fintermedio = DistAcum(N0,intermedio);


%plot(1:N0, y, '.', 'MarkerSize', 20)

function [prob_ini prob_inter] = ProbPoisson(Gamma,N0,dt)
  mu1 = N0*Gamma*dt;
  mu2 = N0*Gamma*30*dt;
  prob_ini = zeros(N0,1); %probabilidad de que un número x de núcleos decaiga en un tiempo inicial
  prob_inter = zeros(N0,1); %probabilidad de que un número x de núcleos decaiga en un tiempo intermedio 
  for i = 1:N0
    prob_ini(i) = (mu1^(i) * exp(-mu1))/factorial(i);
    prob_inter(i) = (mu2^(i) * exp(-mu2))/factorial(i);
  end
  prob_ini = [exp(-mu1); prob_ini];
  prob_inter = [exp(-mu2); prob_inter];
  
  figure;
  plot(0:N0,prob_ini, 'b');
  hold on
  plot(0:N0,prob_inter, 'r');
  title('Probabilidad de que x núcleos decaigan');
  ylabel('Probabilidad');
  xlabel('x núcleos');
  legend('Tiempo Inicial', 'Tiempo Intermedio');
end

function Facum = DistAcum(N0,densidad)
  Facum = zeros(length(densidad),1);
  Facum(1) = densidad(1);
  for i = 1:N0
    Facum(i+1) =  Facum(i) + densidad(i+1)
  end
end
  
  

  
  

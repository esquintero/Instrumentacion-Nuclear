p = (0.7/2.6)*0.2; %Probabilidad binomial de decaer
q = 1-p;
N0=100000000; %Número de núcleos en t=0
dt= 0.2;  %delta t escogido
gamma = 0.7/2.6; %Constante de decamiento del Na-22

grid = (5:dt:15)'; % grilla de tiempos discretizados
bingo=1;

%%%%%Teorico ley de decaimiento

Nt = zeros(length(grid),1);
for i=0:length(grid)-1
    Nt(i+1) = N0*exp(-(0.7/2.6)*i*0.2);
end

%%%%%%Exponencial
fdp = (1:length(grid))';

for i=1:length(grid)
    fdp(i)= -exp(-gamma*(grid(i)+dt)) + exp(-gamma*grid(i));
end

Facum = zeros(length(grid),1);
for i=1:length(grid)-1
    Facum(i+1)= Facum(i) + fdp(i+1);
end
%bingo = 1;
frec = zeros(length(grid),1);
for i=1:N0    
    r = random('uniform',0,1);
    for j=2:length(grid)
        if Facum(j-1) < r && r < Facum(j)
            bingo = j;
            break
        end
    end
    frec(bingo) = frec(bingo) + 1;
end


  
 
%%%%%%Binomial
N=50; %Número de intentos para la acumulada
fdpx1=zeros(N,1);
fdpx0=zeros(N,1); 



for i=1:N
    fdpx1(i) = (factorial(i)/factorial(i-1))*p*q^(i-1); %fdp binomial con x=1 y N=50
    fdpx0(i) = (factorial(i)/factorial(i))*q^(i);       %fdp binomial con x=0 y N=50
end

for i=1:N
    if fdpx1(i) > fdpx0(i)
        Fx(i) = 1;            %Para N intentos, si la fdp(x=1) es mayor que la fdp(x=0), decae
    end
end

%plot(1:N,Fx)
%hold on
%xlabel('Intentos N')
%ylabel('F(x)')
%hold off
aux=N0;
frecuencias = zeros(length(grid),1);
for i=1:length(grid)    
    
    for j=1:aux
        r = random('uniform',0,1);
        if r < q
            frecuencias(i) = frecuencias(i)+1; %Simulación de núcleos que no decaen con la binomial
        end
        
    end
    aux = frecuencias(i);
    
end



M = max(frec);

stairs(grid,frecuencias*M/N0,'b')
hold on
stairs(grid,Nt*M/N0,'r')
title('Con No=10^8')
xlabel('t(años)')
ylabel('N(t)')
stairs(grid,frec,'k')
%axis([0 5 0 M])
legend('Simulación binomial','Ley de decaimiento', 'Simulación exponencial')
hold off



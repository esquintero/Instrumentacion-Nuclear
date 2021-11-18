A = load('amplificador.txt');
M = max(A);
N = min(A);
aux = 0.05*(M-N);
Vmin = N-aux;
Vmax = M+aux;
numbox = (M-N)/2.5;
grid = zeros(48,1);

for i=1:size(grid,1)
    grid(i) = Vmin + 2.5*i;
end

frecuencias = zeros(length(grid),1);

for i=1:47
    frecuencias(i) = sum(grid(i)< A & A < grid(i+1));
end

hold on
bar(grid,frecuencias)

hold off







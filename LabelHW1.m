%% Genero i punti e li etichetto in base alla classe

%numero di punti casuali da generare
punti = 1000;

rng('default'); % For reproducibility

%coppie di punti con coordinate in 2d
X = [(randn(punti,2)*(0.75))+3;
    (randn(punti,2)*1)-2];

%etichettare casualmente il 3% dei dati
y = [ones(punti,1);-ones(punti,1)];
g = lab_mach(y);

%plot dei punti casuali con etichette
figure(1)
gscatter(X(:,1),X(:,2),g);
grid on;
title('Randomly Generated Data');

%numero di punti etichettati
n_lab = sum(abs(g));

%Seleziono i dati in base alla classe di appartenenza
X_lab1 = X(g(:,1) == 1, : );
X_lab2 = X(g(:,1) == -1, : );
X_un = X(g(:,1) == 0, : );

y_lab1 = g(g(:,1) == 1, : );
y_lab2 = g(g(:,1) == -1, : );
y_un = g(g(:,1) == 0, : );

y_un_true = y(g(:,1) == 0, : );

%unisco i dati labled in un unica matrice
X_lab = [X_lab1 ; X_lab2];

y_lab = [y_lab1 ; y_lab2];

%Similarity measure, distanza tra unlabeled e labeled
%w_ij = pdist2(X_lab(1,:),X_un(2,:));
%w = pdist2(X_lab,X_un);
w = exp(-pdist2(X_lab,X_un));

%distanza tra i vari unlabeled 
%w_bar = pdist2(X_un,X_un);
w_bar = exp(-pdist2(X_un,X_un));

%contatore di iterazioni
it = 1;

% Optimality tolerance:
eps = 1.0e-1;

% Stopping criterion
%
% 1 : reach of a target value for the obj.func. fk - fstop <= eps
% 2 : nabla f(xk)'dk <= eps
stopcr = 2;

%verbosity =0 doesn't display info, verbosity =1 display info
verb=1;

%Calcolo della lc come massimo degli autovalori dell'hessiana.
Hess= zeros(length(y_un),length(y_un));

for a=1:length(y_un)
    Hess(:,a)= -2*w_bar(:,a);
    Hess(a,a)= 2*sum(w(:,a))+2*sum(w_bar(:,a))-2*w_bar(a,a);
end

autovalori = eig(Hess);

%Valore della Lipschitz constant calcolato come
%massimo degli autovettori
lc = max(autovalori);
sigma = min(autovalori);

fstop = 10;
maxit = 100;
%l'armijo (arls=1) non funziona
arls=3;

%% Inizio dell 'algoritmo (chiamata alla funzione)

disp('*****************');
disp('*  GM STANDARD  *');
disp('*****************');

%ygm è il vettore delle previsioni prodotte dal metodo (cioè il minimo
%della funzione a cui sono interesato
%itergm è il numero di iterazioni fatte dal metodo

[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
G_descent(w,y_lab,w_bar,y_un,y_un_true,lc,verb,arls,maxit,eps,fstop,stopcr);


%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
%BCGD_rand(w,y_lab,w_bar,y_un,y_un_true,lc,verb,maxit,eps,fstop,stopcr);

%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
%BCGD_cyclic(w,y_lab,w_bar,y_un,y_un_true,lc,verb,maxit,eps,fstop,stopcr);


% Print results:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'f(y)  = %10.3e\n',fxgm);
%fprintf(1,'Number of non-zero components of x = %d\n',...
%   sum((abs(xgm)>=0.000001)));
fprintf(1,'Number of iterations = %d\n',itergm);
fprintf(1,'||gr||^2 = %d\n',gnrgm(maxit));
fprintf(1,'CPU time so far = %10.3e\n', tottimegm);


%plot figure
fmin= min(fhgm);

% Uncomment for better error analysis
%fmin =min(fhgm0);

%% plot dei risultati del metodo

%plot figure
figure(2)
semilogy(timeVecgm,fhgm-fmin,'r-') %usa una scala logaritmica su y 
title('Gradient Method  - objective function')
legend('GM')
%xlim([0,50]); 
xlabel('time'); 
%ylim([10^(-5),10^4]); 
ylabel('err');

%plot figure
figure(3)
semilogy(fhgm-fmin,'r-')
title('Gradient Method  - objective function')
legend('GM')
%xlim([0,10000]); 
xlabel('iter'); 
%ylim([10^(-5),10^4]); 
ylabel('err');


%plot figure accuratezza vs cpu time
figure(4)
plot(timeVecgm,accuracy,'r-') 
title('Gradient Method  - Accuracy')
%legend('GM')
xlim([0,timeVecgm(itergm-1)]); 
xlabel('time'); 
%ylim([10^(-5),10^4]); 
ylabel('accuracy');



% Plot del clustering stimato arrotondando ygm

%gscatter(X_lab(:,1),X_lab(:,2),y_lab);
%grid on;
%title('Predicted clustering');
%hold on
%gscatter(X_un(:,1),X_un(:,2),round(ygm));
%hold off



hvsd = @(x) [0.5*(x == 0) + (x > 0)];


% ygm_rounded = hvsd(ygm);
% Y_true = ones(length(ygm_rounded));
% counter = 0;

%for i = 1:length(ygm_rounded)
%     if (ygm_rounded(i) == Y_true(i))
%         counter = counter + 1;
%     end
% end
%accuracy = counter/length(ygm_rounded);


%% 
%Questo plot dipende dalla funzione di attivazione 
%hvsd e round producono due risultati diversi
figure(5)
gscatter(X_lab(:,1),X_lab(:,2),y_lab);
grid on;
title('Predicted clustering');
hold on
gscatter(X_un(:,1),X_un(:,2),hvsd(ygm)-hvsd(-ygm));
hold off
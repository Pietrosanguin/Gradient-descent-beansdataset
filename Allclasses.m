T = readtable('Dry_Bean_Dataset_All_Classes.xlsx');
T = table2array(T);
R = readtable('Dry_Bean_Dataset_Complete.xlsx');
R = table2array(R);


T(:,1) = [];
X = T(:,1:2);
X_un = T(T(:,3) == 0, 1:2);
X_lab1 = T(T(:,3) == 1, 1:2);
X_lab2 = T(T(:,3) == 2, 1:2);
X_lab3 = T(T(:,3) == 3, 1:2);
X_lab4 = T(T(:,3) == 4, 1:2);
X_lab5 = T(T(:,3) == 5, 1:2);
X_lab6 = T(T(:,3) == 6, 1:2);
X_lab7 = T(T(:,3) == 7, 1:2);
X_lab = [X_lab1 ; X_lab2];

y = T(:,3);
y_lab1 = T(T(:,3) == 1 ,3);
y_lab2 = T(T(:,3) == 2,3);
y_lab3 = T(T(:,3) == 3,3);
y_lab4 = T(T(:,3) == 4,3);
y_lab5 = T(T(:,3) == 5,3);
y_lab6 = T(T(:,3) == 6,3);
y_lab7 = T(T(:,3) == 7,3);
y_lab = [y_lab1 ; y_lab2];
y_un = T(T(:,3) == 0,3);

Y_true = R(:,4);

w = exp(-pdist2(X_lab,X_un));

%distanza tra i vari unlabeled 
%w_bar = pdist2(X_un,X_un);
w_bar = exp(-pdist2(X_un,X_un));


%%
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

%Valore della Lipschitz constant dato a caso, bisogna calcolarlo come
%massimo degli autovettori
lc = max(autovalori);


sigma = min(autovalori);

fstop = 40000;
maxit = 10000;
%l'armijo (arls=1) non funziona
arls=3;

disp('*****************');
disp('*  GM STANDARD  *');
disp('*****************');

%ygm è il vettore delle previsioni prodotte dal metodo (cioè il minimo
%della funzione a cui sono interesato
%itergm è il numero di iterazioni fatte dal metodo

[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm]=...
G_descent(w,y_lab,w_bar,y_un,lc,verb,arls,maxit,eps,fstop,stopcr);


%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm]=...
%BCGD_rand(w,y_lab,w_bar,y_un,lc,verb,maxit,eps,fstop,stopcr);

%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm]=...
%BCGD_cyclic(w,y_lab,w_bar,y_un,lc,verb,maxit,eps,fstop,stopcr);


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

%plot figure
figure
semilogy(timeVecgm,fhgm-fmin,'r-') %usa una scala logaritmica su y 
hold on

title('Gradient Method  - objective function')
legend('GM')
%xlim([0,50]); 
xlabel('time'); 
%ylim([10^(-5),10^4]); 
ylabel('err');

%plot figure
figure
semilogy(fhgm-fmin,'r-')

title('Gradient Method  - objective function')
legend('GM')
%xlim([0,10000]); 
xlabel('iter'); 
%ylim([10^(-5),10^4]); 
ylabel('err');



% Plot del clustering stimato arrotondando ygm

%gscatter(X_lab(:,1),X_lab(:,2),y_lab);
%grid on;
%title('Predicted clustering');
%hold on
%gscatter(X_un(:,1),X_un(:,2),round(ygm));
%hold off



hvsd = @(x) [0.5*(x == 0) + (x > 0)];

ygm_rounded = hvsd(ygm);
Y_true = ones(length(ygm_rounded));
counter = 0;
for i = 1:length(ygm_rounded)
    if (ygm_rounded(i) == Y_true(i))
        counter = counter + 1;
    end
end
accuracy = counter/length(ygm_rounded);


gscatter(X_lab(:,1),X_lab(:,2),y_lab);
grid on;
title('Predicted clustering');
hold on
gscatter(X_un(:,1),X_un(:,2),hvsd(ygm)-hvsd(-ygm));
hold off
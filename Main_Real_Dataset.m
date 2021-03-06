T = readtable('Dry_Bean_Dataset.xlsx');
T = table2array(T);
R = readtable('Dry_Bean_Dataset_Complete.xlsx');
R = table2array(R);
T(:,1) = []; % First column has been removed due to being the index column


X = T(:,1:2);
X_un = T(T(:,3) == 0, 1:2);
X_lab2 = T(T(:,3) == 1, 1:2);
X_lab1 = T(T(:,3) == -1, 1:2);
X_lab = [X_lab1 ; X_lab2];

y = T(:,3);
y_lab1 = T(T(:,3) == -1 ,3);
y_lab2 = T(T(:,3) == 1,3);
y_lab = [y_lab1 ; y_lab2];
y_un = T(T(:,3) == 0,3);

Y_true = R(:,4);
Y_true = Y_true(T(:,3) == 0, : );
gamma = 1000;

% Heavyside function
hvsd = @(x) [0.5*(x == 0) + (x > 0)];

% Distance between labeled & unlabeled
w = exp(- gamma * pdist2(X_lab,X_un).^2);

% Distance between unlabeled & unlabeled 
w_bar =exp(- gamma * pdist2(X_un,X_un).^2);

% Plot of the initial status of the data

gscatter(T(:,1),T(:,2),T(:,3),"rcb");


% Iterations' counter
it = 1;

% Optimality tolerance:
eps = 1.0e-1;

% Stopping criterion
%
% 1 : reach of a target value for the obj.func. fk - fstop <= eps
% 2 : nabla f(xk)'dk <= eps
stopcr = 2;

%verbosity =0 doesn't display info, verbosity =1 displays info
verb=1;

%Computation of  lc as maximum eigenvalue of the hessian matrix
Hess= zeros(length(y_un),length(y_un));

for a=1:length(y_un)
    Hess(:,a)= -2*w_bar(:,a);
    Hess(a,a)= 2*sum(w(:,a))+2*sum(w_bar(:,a))-2*w_bar(a,a);
end

eigenvalues = eig(Hess);
lc = max(eigenvalues);
sigma = min(eigenvalues);


fstop = 10000; % set to preferred value
maxit = 10; % set to preferred value


disp('*****************');
disp('*  GM STANDARD  *');
disp('*****************');

%ygm is the vector of predicted labels returned by the algorithm
%itergm is the number of iterations performed by the algorithm


%%%%%%% CODE FOR RUNNING STANDARD GRADIENT DESCENT ALGORITHM

[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
Gradient_Descent(w,y_lab,w_bar,y_un,Y_true,lc,verb,maxit,eps,fstop,stopcr);


%%%%%%% CODE FOR RUNNING BCGD RANDOMIZED GRADIENT DESCENT ALGORITHM

%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
%BCGD_Randomized(w,y_lab,w_bar,y_un,Y_true,lc,verb,maxit,eps,fstop,stopcr);

%%%%%%% CODE FOR RUNNING BCGD CYCLIC GRADIENT DESCENT ALGORITHM

%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
%BCGD_Cyclic(w,y_lab,w_bar,y_un,Y_true,lc,verb,maxit,eps,fstop,stopcr);


% Print results:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'f(y)  = %10.3e\n',fxgm);
fprintf(1,'Number of iterations = %d\n',itergm);
fprintf(1,'||gr||^2 = %d\n',gnrgm(maxit));
fprintf(1,'CPU time so far = %10.3e\n', tottimegm);

fmin= min(fhgm);


% Plot of the results

% Plot cpu time vs err
figure(2)
semilogy(timeVecgm,fhgm-fmin,'r-') 
xlabel('cpu time (s)'); 
ylabel('err');


% Plot iter vs err
figure(3)
semilogy(fhgm-fmin,'r-') 
xlabel('iter');  
ylabel('err');


% Plot cpu time vs accuracy 
figure(4)
plot(timeVecgm,accuracy,'r-') 
xlim([0,timeVecgm(itergm-1)]); 
xlabel('cpu time (s)');  
ylabel('accuracy');


% Plot predicted clustering
figure(5)
gscatter(X_lab(:,1),X_lab(:,2),y_lab,"rcb");
grid on;
title('Predicted clustering');
hold on
gscatter(X_un(:,1),X_un(:,2),hvsd(ygm)-hvsd(-ygm),"rcb");
hold off
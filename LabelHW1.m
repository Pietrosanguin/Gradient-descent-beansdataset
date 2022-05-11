% Number of random points to be generated
punti = 5000;

rng('default'); % For reproducibility

% Randomly generated points with 2d coordinates
X = [(randn(punti,2)*(0.75))+3;
    (randn(punti,2)*(1))-2];

% Random tagging of 3% the data
y = [ones(punti,1);-ones(punti,1)];
g = lab_mach(y);

% Plot of random data with tags
figure(1)
gscatter(X(:,1),X(:,2),g,"rcb");
grid on;
title('Randomly Generated Data');

% Heavyside function
hvsd = @(x) [0.5*(x == 0) + (x > 0)];

% Number of tagged points
n_lab = sum(abs(g));

% Selection of the data with respect to the class
X_lab1 = X(g(:,1) == 1, : );
X_lab2 = X(g(:,1) == -1, : );
X_un = X(g(:,1) == 0, : );

y_lab1 = g(g(:,1) == 1, : );
y_lab2 = g(g(:,1) == -1, : );
y_un = g(g(:,1) == 0, : );

y_un_true = y(g(:,1) == 0, : );

% Join of labeled data in an unique matrix
X_lab = [X_lab1 ; X_lab2];

y_lab = [y_lab1 ; y_lab2];

% Distance between labeled and unlabeled

w = exp(-pdist2(X_lab,X_un).^2);

% Distance between unlabeled and unlabeled

w_bar = exp(-pdist2(X_un,X_un).^2);

% Iteration counter
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

% Computation of lc as max eigenvalue of the Hessian
Hess= zeros(length(y_un),length(y_un));

for a=1:length(y_un)
    Hess(:,a)= -2*w_bar(:,a);
    Hess(a,a)= 2*sum(w(:,a))+2*sum(w_bar(:,a))-2*w_bar(a,a);
end

eigenvalues = eig(Hess);
lc = max(eigenvalues);
sigma = min(eigenvalues);

fstop = 10000; % set to preferred value
maxit = 10000; % set to preferred value

% Start of algorithm

disp('*****************');
disp('*  GM STANDARD  *');
disp('*****************');

%ygm is the vector of predicted labels returned by the algorithm
%itergm is the number of iterations performed by the algorithm

[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
G_descent(w,y_lab,w_bar,y_un,y_un_true,lc,verb,maxit,eps,fstop,stopcr);

%%%%%%% CODE FOR RUNNING BCGD RANDOMIZED GRADIENT DESCENT ALGORITHM

%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
%BCGD_rand(w,y_lab,w_bar,y_un,y_un_true,lc,verb,maxit,eps,fstop,stopcr);

%%%%%%% CODE FOR RUNNING BCGD CYCLIC GRADIENT DESCENT ALGORITHM

%[ygm,itergm,fxgm,tottimegm,fhgm,timeVecgm,gnrgm,accuracy]=...
%BCGD_cyclic(w,y_lab,w_bar,y_un,y_un_true,lc,verb,maxit,eps,fstop,stopcr);


% Print results:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(1,'f(y)  = %10.3e\n',fxgm);
fprintf(1,'Number of iterations = %d\n',itergm);
fprintf(1,'||gr||^2 = %d\n',gnrgm(maxit));
fprintf(1,'CPU time so far = %10.3e\n', tottimegm);

fmin= min(fhgm);

% Plots of the results


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
xlim([0,timeVecgm(itergm)]); 
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
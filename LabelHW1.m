%Codice utile per implementare un risultato simile a quello che voglio
%https://it.mathworks.com/matlabcentral/answers/461354-label-my-data-set-automatically-and-group-the-similar-points-or-the-nearest-points-with-the-same

%Ci sono due modi per generare i clusters: gaussianamente oppure tramite
%linkage, NB controllare il dendrogram (cluster con numeri casuali)

%provo con il linkage
%numero di punti casuali da generare
punti = 1000;

rng('default'); % For reproducibility

%coppie di punti con coordinate in 2d
X = [(randn(punti,2)*(0.75))+3;
    (randn(punti,2)*1)-2];

%etichettare casualmente il 3% dei dati
y = [ones(punti,1);-ones(punti,1)];
g=lab_mach(y);

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

%unisco i dati labled in un unica matrice
X_lab = [X_lab1 ; X_lab2];

y_lab = [y_lab1 ; y_lab2];


%Considero come similarity measure la distanza euclidea (volendo si può
%cambiare in minkowski)

%distanza tra unlabeled e labeled
%w_ij = pdist2(X_lab(1,:),X_un(2,:));
w_ij = pdist2(X_lab,X_un);

%distanza tra i vari unlabeled 
w_bar_ij = pdist2(X_un,X_un);

%calcolo il gradiente rispetto agli unlabeled componente per componente


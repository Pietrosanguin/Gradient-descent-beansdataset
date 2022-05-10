function [y_un,it,fy,ttot,fh,timeVec,gnrit,accuracy] = G_descent(w,y_lab,w_bar,y_un,y_un_true,lc,verbosity,maxit,eps,fstop,stopcr)

% Parametri di output della funzione: 
%x:  è il punto di minimo a cui vogliamo convergere
%it:  è il l'iterazione a cui siamo arrivati
%fx:  è il valore della funzione aggiornato ad ogni iterazione
%ttot: contiene il cpu time totale 
%fh: è un array che contiene il valore della funzione all'iterazione it
%timeVec: è un array che contiene il cpu time ad ogni iterazione
%gnrit: è la norma del gradiente ad ogni iterazione it
%y_un_true è il vettore con le classi vere dei punti y_un
%
% Parametri di input
%y_un che do' in input è lo starting point
%w_ij, w_bar_ij, y:un sono parametri del problema
%lc: è la costante di Lipschitz (poi vedremo come determinarla)
%arls: è il tipo di Armijo line search che può essere 1 2 o 3 per il momento ho guardato solo 1 e 3 
%verbosity: è un parametro che se >0 mostra cosa fa l'algoritmo ad ogni iterazione

hvsd = @(x) [0.5*(x == 0) + (x > 0)];
        
%gamma=0.0001;
maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
accuracy=zeros(1,maxit);
flagls=0;

tic; %fa partire il calcolo del tempo
timeVec(1) = 0;

%Values for the smart computation of the o.f. 
%Calcolo le due parti della funzione da minimizzare e poi le sommo tra loro
%sum1 e sum2 sono la somma di tre termini che escono dallo sviluppo dei
%quadrati

fy = 10000;

it=1;
alpha=1/lc;

first_term = zeros(length(y_un),1);
second_term = zeros(length(y_un),1);
g = zeros(length(y_un),1);


while (flagls==0)
    %vectors updating
    if (it==1)
        timeVec(it) = 0;
    else
        timeVec(it) = toc;
    end
    fh(it)=fy;
    
    % gradient evaluation
    
    for j= 1:length(y_un)
        
            first_term(j) = 2*(sum(w(:,j))*y_un(j))-2*transpose(y_lab)*w(:,j);
    end
    
    for j = 1:length(y_un)
         
           second_term(j) = 2*sum(w_bar(:,j))*y_un(j)-2*transpose(y_un)*w_bar(:,j);
 
    end
     
    for j = 1:length(y_un)
    
       g(j) = first_term(j) + second_term(j); 
    
    end
    
    %gnr is the gradient's norm, gnrit saves its value at each iteration

    gnr = norm(g);
    gnrit(it) = -gnr;
        
        % stopping criteria and test for termination
    if (it>=maxniter)
        break;
    end

    switch stopcr  
        case 1
            % continue if not yet reached target value fstop
            if (fy<=fstop)
                break
            end
        case 2
            % stopping criterion based on the product of the 
            % gradient with the direction
            if (abs(gnr) <= eps)
                break;
            end
        otherwise
            error('Unknown stopping criterion');
    end % end of the stopping criteria switch
    
         
    y_un=y_un-alpha*g;
    
    sum1 = 0;
    sum2 = 0;

    for i = 1:length(y_lab)
        for j = 1:length(y_un)
            
           sum1 = sum1 + w(i,j)*((y_un(j)-y_lab(i))^2);

        end
    end

    for i = 1:length(y_un)
        for j = 1:length(y_un)
            
           sum2 = sum2 + w_bar(i,j)*((y_un(i)-y_un(j))^2);

        end
    end

    fy=sum1 + 0.5*sum2;
   
    accuracy(it) = sum(y_un_true == hvsd(y_un)-hvsd(-y_un),'all')/numel(y_un);
       
    if (verbosity>0)
        disp(['-----------------** ' num2str(it) ' **------------------']);
        disp(['gnr      = ' num2str(abs(gnr))]);
        disp(['f(y)     = ' num2str(fy)]);
        disp(['alpha     = ' num2str(alpha)]);
        disp(['accuracy max    = ' num2str(max(accuracy))]); 
    end
                 
    it = it+1;
    
    if(it<maxit)
        fh(it+1:maxit)=fh(it);
        gnrit(it+1:maxit)=gnrit(it);
        timeVec(it+1:maxit)=timeVec(it);
    end


    ttot = toc;


end
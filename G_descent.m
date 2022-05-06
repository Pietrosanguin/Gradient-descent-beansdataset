function [y_un,it,fx,ttot,fh,timeVec,gnrit] = G_descent(w,y_lab,w_bar,y_un,lc,verbosity,arls,maxit,eps,fstop,stopcr)

% Parametri di output della funzione: 
%x:  è il punto di minimo a cui vogliamo convergere
%it:  è il l'iterazione a cui siamo arrivati
%fx:  è il valore della funzione aggiornato ad ogni iterazione
%ttot: contiene il cpu time totale 
%fh: è un array che contiene il valore della funzione all'iterazione it
%timeVec: è un array che contiene il cpu time ad ogni iterazione
%gnrit: è la norma del gradiente ad ogni iterazione it
%
% Parametri di input
%y_un che do' in input è lo starting point
%w_ij, w_bar_ij, y:un sono parametri del problema
%lc: è la costante di Lipschitz (poi vedremo come determinarla)
%arls: è il tipo di Armijo line search che può essere 1 2 o 3 per il momento ho guardato solo 1 e 3 
%verbosity: è un parametro che se >0 mostra cosa fa l'algoritmo ad ogni iterazione

%global lc;
        
gamma=0.0001;
maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
flagls=0;

tic; %fa partire il calcolo del tempo
timeVec(1) = 0;

fx = 0;

it=1;

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
    fh(it)=fx;
    
    % gradient evaluation, ho sostituito il nostro gradiente con quello
    % calcolato da lui nel vecchio script

    for j= 1:length(y_un)
        for i = 1:length(y_lab)
    
            first_term(j) = 2*(w(i,j)*(y_un(j)-y_lab(i)));
             
        end
    end
    
    for j = 1:length(y_un)
        for i = 1:length(y_un)
         
           second_term(j) = 2*(w_bar(i,j)*(y_un(j)-y_un(i)));
    
        end
    end
     
    for j = 1:length(y_un)
    
       g(j) = first_term(j) + second_term(j); 
    
    end

    d=-g;
    
    %gnr è la norma del gradiente, viene salvata nel vettore gnrit per ogni
    %iterazione it
    gnr = g'*d;
    gnrit(it) = -gnr;
        
        % stopping criteria and test for termination
    if (it>=maxniter)
        break;
    end
        switch stopcr  
            case 1
                % continue if not yet reached target value fstop
                if (fx<=fstop)
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
        
        switch arls
            case 1
                %Armijo search

            case 2
                %exact alpha

            otherwise
                %fixed alpha
                alpha = 1/lc;
                y_un = y_un + alpha*d;
                
                sum1z = 0;
                sum2z = 0;

                for i = 1:length(y_lab)
                    for j = 1:length(y_un)
                        
                       sum1z = sum1z + w(i,j)*((y_un(j)-y_lab(i))^2);

                    end
                end

                for i = 1:length(y_un)
                    for j = 1:length(y_un)
                        
                       sum2z = sum2z + w_bar(i,j)*((y_un(i)-y_un(j))^2);

                    end
                end

                fx=sum1z + 0.5*sum2z;
              
        end
        
        
        if (verbosity>0)
            disp(['-----------------** ' num2str(it) ' **------------------']);
            disp(['gnr      = ' num2str(abs(gnr))]);
            disp(['f(x)     = ' num2str(fx)]);
            disp(['alpha     = ' num2str(alpha)]);                    
        end
                     
        it = it+1;
        
        
end

if(it<maxit)
    fh(it+1:maxit)=fh(it);
    gnrit(it+1:maxit)=gnrit(it);
    timeVec(it+1:maxit)=timeVec(it);
end


ttot = toc;


end

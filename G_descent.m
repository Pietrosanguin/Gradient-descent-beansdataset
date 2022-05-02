 function [x,it,fx,ttot,fh,timeVec,gnrit] = G_descent(w,y_lab,w_bar,y_un,lc,verbosity,arls,maxit,eps,fstop,stopcr)

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

global lc;
        
gamma=0.0001;
maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
flagls=0;

tic; %fa partire il calcolo del tempo
timeVec(1) = 0;

%Values for the smart computation of the o.f. 
%Calcolo le due parti della funzione da minimizzare e poi le sommo tra loro
%sum1 e sum2 sono la somma di tre termini che escono dallo sviluppo dei
%quadrati

sum1 = sum(w*(y_un.^2))+(y_lab.^2).'*sum(w,2)-2*y_lab.'*(w*y_un);
sum2 = sum(w_bar*(y_un.^2))+(y_un.^2).'*sum(w_bar,2)-2*y_un.'*(w_bar*y_un);

fx=sum1 + 0.5*sum2;   %nella notazione del prof è fx nel nostro caso sarebbe fy


it=1;


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
     
    first_term = zeros(length(y_un),1);
    second_term = zeros(length(y_un),1);
    g = zeros(length(y_un),1);

    for j= 1:length(y_un)
        for i = 1:length(y_lab)
    
            first_term(j) = 2*(w(i,j)*(y_un(j)-y_lab(i)));
             
        end
    end
    
    for j = 1:length(y_un)
        for i = 1:length(y_un)
         
           second_term(j) = 2*w_bar(i,j)*(y_un(j)-y_un(i));
    
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
        
        
        
        %linesearch
        %arls=2;
        
        switch arls
            case 1
                 %Armijo search
            alpha=3;
            ref = gamma*gnr;
            
            while(1)
                z=y_un+alpha*d;
                %Computation of the o.f. at the trial point
                sum1z = sum(w*(z.^2))+(y_lab.^2).'*sum(w,2)-2*y_lab.'*(w*z);
                sum2z = sum(w_bar*(z.^2))+(z.^2).'*sum(w_bar,2)-2*z.'*(w_bar*z);

                fz=sum1z + 0.5*sum2z;
                         
                if (fz<=fx+alpha*ref)
                    z = y_un + alpha*d;
                    break;
                else
                    alpha=alpha*0.1;
                end
                
                if (alpha <= 1e-20)
                    z=y_un;
                    fz=fx;
                    flagls=1;
                    it = it-1;
                    break;
                end
                
            end
            case 2
                %exact alpha
                alpha=-gnr/((d'*Q)*d);
                z=x+alpha*d;
                Qz = Q*z;
                zQz= z'*Qz;
                cz = c'*z;
                fz = 0.5*zQz-cz;
            otherwise
               %fixed alpha
                alpha=1/lc;
                z=y_un+alpha*d;
                sum1z = sum(w*(z.^2))+(y_lab.^2).'*sum(w,2)-2*y_lab.'*(w*z);
                sum2z = sum(w_bar*(z.^2))+(z.^2).'*sum(w_bar,2)-2*z.'*(w_bar*z);
                
                for j= 1:length(z)
                gz = transpose(2*(z(j)-y_lab).'*w+2*(z(j)-y_un).'*w_bar);
    
                end 

                fz=sum1z + 0.5*sum2z;
              
        end

        y_un=z;
        g = gz;
        fx = fz;
        
        
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

function [x,it,fx,ttot,fh,timeVec,gnrit] = BCGD_rand(w,y_lab,w_bar,y_un,lc,verbosity,maxit,eps,fstop,stopcr)
%BCGD_RAND La funzione implementa il BCGD method con randomized block
%coordinate

%input: 
%i blocchi devono avere dimensione 1 b=length(y_un), perciò ho implementato
%direttamente la matrice identità senza passare b alla funzione
%y_un variabile per cui minimizzo

%definisco la matrice di blocchi identità
U = eye(length(y_un));

%I blocchi hanno dimensione uno perciò b è:
b = length(y_un);


fx = 0;
it=1;


maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
flagls=0;

first_term = zeros(length(y_un),1);
second_term = zeros(length(y_un),1);
g = zeros(length(y_un),1);

tic; %fa partire il calcolo del tempo
timeVec(1) = 0;


while (flagls==0)
    %vectors updating
    if (it==1)
        timeVec(it) = 0;
    else
        timeVec(it) = toc;
    end
    fh(it)=fx;
    
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
        
        %Qui inizia l'algoritmo BCGD randomized
        
        %Considero una distribuzione uniforme da cui estrarre casualmente
        %il blocco
               %fixed alpha
                alpha=1/lc;
                ik = randi([1 b],1); %estrae uniformemente un numero tra 1 e b
                z=y_un+alpha*U(:,ik)*d(ik);
                %sum1z = sum(w*(z.^2))+(y_lab.^2).'*sum(w,2)-2*y_lab.'*(w*z);
                %sum2z = sum(w_bar*(z.^2))+(z.^2).'*sum(w_bar,2)-2*z.'*(w_bar*z);
                
                sum1z = 0;
                sum2z = 0;

                for i = 1:length(y_lab)
                    for j = 1:length(y_un)
                        
                       sum1z = sum1z + w(i,j)*((z(j)-y_lab(i))^2);

                    end
                end

                for i = 1:length(y_un)
                    for j = 1:length(y_un)
                        
                       sum2z = sum2z + w_bar(i,j)*((z(i)-z(j))^2);

                    end
                end

                fz=sum1z + 0.5*sum2z;
                
                %for j= 1:length(z)
                %    gz = transpose(2*(z(j)-y_lab).'*w+2*(z(j)-y_un).'*w_bar);
    
                %end 
       
        y_un=z;
        fx = fz;
       
        
        
        if (verbosity>0)
            disp(['-----------------** ' num2str(it) ' **------------------']);
            disp(['gnr      = ' num2str(abs(gnr))]);
            disp(['f(y)     = ' num2str(fx)]);
            disp(['alpha     = ' num2str(alpha)]);                    
        end
                     
        it = it+1;
        
        
end

x=z;

if(it<maxit)
    fh(it+1:maxit)=fh(it);
    gnrit(it+1:maxit)=gnrit(it);
    timeVec(it+1:maxit)=timeVec(it);
end


ttot = toc;

end















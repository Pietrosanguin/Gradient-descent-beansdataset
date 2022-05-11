function [y_un,it,fy,ttot,fh,timeVec,gnrit,accuracy] = BCGD_Randomized(w,y_lab,w_bar,y_un,y_un_true,lc,verbosity,maxit,eps,fstop,stopcr)

it=1;

hvsd = @(x) [0.5*(x == 0) + (x > 0)];

maxniter=maxit;
fh=zeros(1,maxit);
gnrit=zeros(1,maxit);
timeVec=zeros(1,maxit);
flagls=0;
accuracy=zeros(1,maxit);

first_term_it = 0;
second_term_it = 0;
g_it = 0;

tic;
timeVec(1) = 0;

first_term = zeros(length(y_un),1);
second_term = zeros(length(y_un),1);
g = zeros(length(y_un),1);

for j= 1:length(y_un)
        
    first_term(j) = 2*(sum(w(:,j))*y_un(j))-2*transpose(y_lab)*w(:,j);
end
    
for j = 1:length(y_un)
         
    second_term(j) = 2*sum(w_bar(:,j))*y_un(j)-2*transpose(y_un)*w_bar(:,j);
 
end
     
for j = 1:length(y_un)
    
    g(j) = first_term(j) + second_term(j); 
    
end

alpha=1/lc;
while (flagls==0)
    
    % Vectors updating
    
    if (it==1)
        timeVec(it) = 0;
    else
        timeVec(it) = toc;
    end
    
    % GRADIENT EVALUATION
    
    % Random selection of a block
    reset(RandStream.getGlobalStream,sum(100*clock));
    ik=randi(length(y_un),1);
    
    % Computation of the gradient only in the randomized ik entry
    first_term_it = 2*(sum(w(:,ik))*y_un(ik))-2*transpose(y_lab)*w(:,ik);
    second_term_it = 2*sum(w_bar(:,ik))*y_un(ik)-2*transpose(y_un)*w_bar(:,ik);
    g_it = first_term_it + second_term_it;  

    % Gradient update in the updated position ik
    g(ik) = g_it;
    
    temp = zeros(length(y_un),1); 
    temp(ik) = g_it; 
    y_un = y_un-alpha*temp;
    
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
    fh(it)=fy;

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
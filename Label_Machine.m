function [y_out] = Label_Machine(y_in)
%lab_mach funzione etichettatrice che prende un vettore e ne etichetta
%casualmente circa il 3% dei dati
%   Detailed explanation goes here

y_out = zeros(length(y_in),1);

rng('default'); % For reproducibility
for i = 1:length(y_in)
    x = rand;
    if x <= 0.03
        y_out(i) = y_in(i);
               
        
        
    end
          






end


end

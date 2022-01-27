function label = assignLabel(A,x,b)
% simple label assignation for single column

[n,d] = size(A);
label = zeros(n,1);

for i = 1:n
    a = A(i,:);
    p0 = 1/(1+exp(a*x));
    p1 = 1/(1+exp(-a*x));
    
    % determine likelihood
    if (p0 > p1)
        label(i) = 0;
    else
        label(i) = 1;
    end
    
    % compare performance with given b
    if (label(i) == b(i))
        label(i) = 1;
    else
        label(i) = 0;
    end

end


end


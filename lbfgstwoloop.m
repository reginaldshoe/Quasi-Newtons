function r = lbfgstwoloop(q,m,svec,yvec,H,k)
% two loop recursion to determine new direction for L-BFGS
% q = gradient as per current x_k
% m = memory size
% svec = vector of size m(or k if k<m) with value s
% yvec = vector of size m(or k if k<m) with value y
% H = current inverse Hessian at step k
% k = current L-BFGS step
if (k < m)
    m = k;
end

alpha = zeros(m,1); %storage of alpha
rho = zeros(m,1);
for i = m:1
    rho(i) = 1/(yvec(:,i)'*svec(:,i));
    alpha(i) = rho(i)*(svec(:,i)'*q);
    q = q-alpha(i)*yvec(:,i);
end
r = H*q;
for i = 1:m
    beta = rho(i)*(yvec(:,i)'*r);
    r = r + (alpha(i) - beta)*svec(:,i);
end
r = -r;
end
    
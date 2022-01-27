function [freg,greg,hreg] = reg(z,lam)
% basic regularization function for (lambda)*(||x||^2)/2
    freg = lam*0.5*(norm(z)^2);
    greg = lam*z;
    hreg = @(v) lam*v; 
end
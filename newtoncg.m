function [ncgx,ncgF,ncgG,ncgk,ncgt,ncglabel] = newtoncg(x,A,b,lambda,alpha,beta,p,maxit,tol)
% Use Newton-CG method with Armijo Line Search to calculate Ax = b 
% Native preconditioned-CG used to calc CG with 100 maxiter and 0.01
% tolerance
% A,x,b are the classic Ax=b values (A=data,x=weights,b=linear_combination)
% alpha is starting value for line search (typically alpha = 1)
% lambda is the regularization rate
% beta is the tuning parameter for Armijo
% maxit is the maximum iterations for GD
% tolerance is the target value for norm(G)

%returns
% ncgx = returns column vector x as at completion of newton-CG 
% ncgF = function value of SoftMax function at completion of newton-CG
% ncgG = Gradient vector of SoftMax function at completion of newton-CG
% ncgk = vector 1:number of iterates to complete newton-CG
% ncgt = time taken to complete newton-CG
% ncglabel = column vector with label [1,0]

[~,d] = size(A);
I = eye(d);
[F,G,~] = softMaxFun(x,A,b,@(x) reg(x,lambda));
ncgF = [F];
ncgG = [norm(G)];
ncgt = [0];
ncglabel = [0];

for k = 1:maxit
    tic;
    [F,G,Hv] = softMaxFun(x,A,b,@(x) reg(x,lambda));
    p = pcg(Hv,-G,0.01,100,I,I,p);
    done = false;
    while (~done)
        x_new = x+alpha*p;
        [Fxk,~,~] = softMaxFun(x_new,A,b,@(x_new) reg(x_new,lambda));
        if (Fxk<=F+beta*alpha*p'*p)
            done = true;
            break
        else
            alpha = alpha*0.5;
        end    
    end
    x = x + alpha*p;
    
    if norm(G) < tol
        fprintf('Newton-CG converged\n')
        break
    end
    t = toc;
    ncg_test = assignLabel(A,x,b);
    ncglabel(k+1) = sum(ncg_test(:) > .99)/length(ncg_test);
    ncgF(k+1) = F;
    ncgG(k+1) = norm(G);
    ncgt(k+1) = ncgt(k) + t;
end
fprintf('terminated at ||G|| = %g, obtained after %g iterations \n', norm(G), k);
ncgx = x;
ncgk = k;

end

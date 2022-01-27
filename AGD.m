function [agx,agF,agG,agk,agt,aglabel] = AGD(x,A,b,lambda,tol,maxit)
% Nesterov accelerated GD (AGD), 
% A,x,b are the classic Ax=b values (A=data,x=weights,b=linear_combination)
% lambda is the regularization rate
% maxit is the maximum iterations for AGD
% tolerance is the target value for norm(G)

%returns
% agx = returns column vector x as at completion of AGD
% agF = function value of SoftMax function at completion of AGD
% agG = Gradient vector of SoftMax function at completion of AGD
% agk = vector 1:number of iterates to complete AGD
% agt = time taken to complete AGD
% aglabel = column vector with label [1,0]

Lg = (0.25*norm(A)^2)+lambda;
kappa = cond(A);
[F,G,~] = softMaxFun(x,A,b,@(x) reg(x,lambda));
xk = x;
x_ = x;
y = x;
agF = [F];
agG = [norm(G)];
agt = [0];
aglabel = [0];
for k = 1:maxit
    tic;
    y = xk + ((sqrt(kappa)-1)/(sqrt(kappa)+1))*(xk-x_);
    [~,Gy,~] = softMaxFun(y,A,b,@(y) reg(y,lambda));
    x_ = xk;
    xk = y - Gy/Lg;
    [Fx,Gx,~] = softMaxFun(xk,A,b,@(xk) reg(xk,lambda));
    if norm(Gx) < tol
        fprintf('AGD converged\n')
        break
    end
    t = toc;
    ag_test = assignLabel(A,xk,b);
    aglabel(k+1) = sum(ag_test(:) > .99)/length(ag_test);
    agF(k+1) = Fx;
    agG(k+1) = norm(Gx);
    agt(k+1) = agt(k) + t;
end
fprintf('AGD terminated at ||G|| = %g, obtained after %g iterations \n', norm(G), k);

agx = xk;
agk = k;
end
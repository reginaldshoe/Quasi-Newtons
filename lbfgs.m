function [lbx,lbF,lbG,lbk,lbt,lblabel] = lbfgs(x,A,b,lambda,alpha0,beta,curve,tol,m,maxit)
% Limited Memory BFGS using Strong Wolfe Line Search Conditions
% A,x,b are the classic Ax=b values (A=data,x=weights,b=linear_combination)
% alpha is starting value for line search (typically alpha = 1)
% curve is the curvature condition for line search
% lambda is the regularization rate
% beta is the tuning parameter for Armijo
% maxit is the maximum iterations for GD
% tolerance is the target value for norm(G)

%returns
% lbx = returns column vector x as at completion of AGD
% lbF = function value of SoftMax function at completion of AGD
% lbG = Gradient vector of SoftMax function at completion of AGD
% lbk = vector 1:number of iterates to complete AGD
% lbt = time taken to complete AGD
% lblabel = column vector with label [1,0]

n = length(x);
svec = zeros(n,m);
yvec = zeros(n,m);
I = eye(n);
[F,G,~] = softMaxFun(x,A,b,@(x) reg(x,lambda));
lbF = [F];
lbG = [norm(G)];
lbt = [0];
lblabel = [0];
done=false;
k=0;
while (~done)
    tic;
    [F,G,~] = softMaxFun(x,A,b,@(x) reg(x,lambda));
    if (k == 0)
        p = -G; % initial direction
    else
        H = ((s'*y)/(y'*y))*I;
        p = lbfgstwoloop(G,m,svec,yvec,H,k); % new direction using two-loop
    end
    
    [alpha,~] = lineSearchWolfeStrong(@(v) softMaxFun(v,A,b,@(v) reg(v,lambda)), x, p, alpha0, beta, curve, 1000);
    x_ = (x + alpha*p);
    s = x_ - x;
    [~,G_,~] = softMaxFun(x_,A,b,@(x_) reg(x_,lambda));
    y = G_ - G;
    
    if k < m
        svec(:,k+1) = s;
        yvec(:,k+1) = y;
    else %drop first col and append new s & y once memory m exceeded
        svec = horzcat(svec(:,2:m),s);
        yvec = horzcat(yvec(:,2:m),y);
    end
       
    x=x_;
    k=k+1;
    
    if norm(G) < tol
        fprintf('L-BFGS converged\n');
        done = true;
        break
    end
    
    if (k>=maxit)
        fprintf("L-BFGS exceeded max iterations\n");
        done = true;
        break
    end
    t = toc;
    lb_test = assignLabel(A,x,b);
    lblabel(k+1) = sum(lb_test(:) > .99)/length(lb_test);
    lbF(k+1) = F;
    lbG(k+1) = norm(G);
    lbt(k+1) = lbt(k) + t;
end
fprintf('terminated at ||G|| = %g, obtained after %g iterations \n', norm(G), k);
lbx = x;
lbk = k;
end
    
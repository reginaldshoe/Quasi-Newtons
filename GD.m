function [gdx,gdF,gdG,gdk,gdt,gdlabel] = GD(x,A,b,lambda,alpha,beta,maxit,tol)
% Classic Gradient descent algorithm with Armijo backtracking line search
% A,x,b are the classic Ax=b values (A=data,x=weights,b=
% alpha is starting value for line search (typically alpha = 1)
% lambda is the regularization rate
% beta is the tuning parameter for Armijo
% maxit is the maximum iterations for GD
% tolerance is the target value for norm(G)

%returns
% gdx = returns column vector x as at completion of GD 
% gdF = function value of SoftMax function at completion of GD
% gdG = Gradient vector of SoftMax function at completion of GD
% gdk = vector 1:number of iterates to complete GD
% gdt = time taken to complete GD
% gdlabel = column vector with label [1,0]

[n,d] = size(A);
[F,G,~] = softMaxFun(x,A,b,@(x) reg(x,lambda));
gdF = [F];
gdG = [norm(G)];
gdt = [0];
gdlabel = [0];

for k = 1:maxit
    tic
    [F,G,~] = softMaxFun(x,A,b,@(x) reg(x,lambda));
    p = -G; %initial direction
    done = false;
    %Armijo backtracking line search to find alpha
    while (~done)
        x_new = x+alpha*p;
        [Fxk,~,~] = softMaxFun(x_new,A,b,@(x) reg(x,lambda));
        if (Fxk<=F-beta*alpha*norm(G)^2)
            done = true;
            break
        else
            alpha = alpha*0.5;
        end    
    end
    x = x + alpha*p;

    if norm(G) < tol
        fprintf('GD converged\n')
        break
    end
    t = toc;
    gd_test = assignLabel(A,x,b);
    gdlabel(k+1) = sum(gd_test(:) > .99)/length(gd_test);
    gdF(k+1) = F;
    gdG(k+1) = norm(G);
    gdt(k+1) = gdt(k) + t;
end

fprintf('GD terminated at ||G|| = %g, obtained after %g iterations \n', norm(G), k);

gdx = x;
gdk = k;
end
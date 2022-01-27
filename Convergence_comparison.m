clear
%Main function to test convergence behaviour for various quasi-newton
%algorithms. Tests objective value/convergence/accuracy vs Iterations/time 
%data from https://archive.ics.uci.edu/ml/datasets/spambase
%

% datasets
[A_train,b_train,A_test,b_test] = loadData;
[~,d] = size(A_train);
b_test = b_test(:,1); %superfluous second column

%global parameters
lambda = 1;     % Regularization rate
maxit = 1000;   % max iterations for each algorithm
tol = 10e-4;    % convergence tolerance 
armijo_beta = 10e-4; % Armijo line search Beta parameter
curvature = 0.9; % Strong Wolfe Lilne Search curvature parameter
x = zeros(1,d)'; % initial column vector 
p = zeros(1,d)'; % initial direction vector

%[objVal,vec f(x),vec grad(x), iterations,vec time,vec label]

%Gradient Descent with Armijo

L_g = (0.25*norm(A_train)^2)+lambda; %Lipschitz bound
alpha = 10/L_g;
[gdx,gdF,gdG,gdk,gdt,~] = GD(x,A_train,b_train,lambda,alpha,armijo_beta,maxit,tol);
[~,~,~,t_gdk,t_gdt,t_gdlab] = GD(x,A_test,b_test,lambda,alpha,armijo_beta,maxit,tol);

%L-BFGS w/ Strong Wolfe Line Search

alpha = 1;
m = 20; %history size
[lbx,lbF,lbG,lbk,lbt,~] = lbfgs(x,A_train,b_train,lambda,alpha,armijo_beta,curvature,tol,m,maxit);
[~,~,~,t_lbk,t_lbt,t_lblab] = lbfgs(x,A_test,b_test,lambda,alpha,armijo_beta,curvature,tol,m,maxit);

%Newton-CG w/ Armijo Line Search

alpha = 1;
[ncgx,ncgF,ncgG,ncgk,ncgt,~] = newtoncg(x,A_train,b_train,lambda,alpha,armijo_beta,p,maxit,tol);
[~,~,~,t_ncgk,t_ncgt,t_ncglab] = newtoncg(x,A_test,b_test,lambda,alpha,armijo_beta,p,maxit,tol);

%Accelerated GD

[agx,agF,agG,agk,agt,~] = AGD(x,A_train,b_train,lambda,tol,maxit);
[~,~,~,t_agk,t_agt,t_aglab] = AGD(x,A_test,b_test,lambda,tol,maxit);


%Plots
figure('Name','||G|| vs Iterations');
plot1 = loglog(1:length(gdG),gdG,'b-','LineWidth',2);
hold on
plot1 = loglog(1:length(ncgG),ncgG,'r-','LineWidth',2);
plot1 = loglog(1:length(lbG),lbG,'g-','LineWidth',2);
plot1 = loglog(1:length(agG),agG,'m-','LineWidth',2);
title('||G|| vs Iterations')
xlabel('Total iterations');
ylabel('Norm of Gradient');
legend('GD','Newton-CG','L-BFGS','AGD','Location','southeast');
hold off

figure('Name','ObjVal vs Iterations');
plot2 = loglog(1:length(gdF),gdF,'b-','LineWidth',2);
hold on
plot2 = loglog(1:length(ncgF),ncgF,'r-','LineWidth',2);
plot2 = loglog(1:length(lbF),lbF,'g-','LineWidth',2);
plot2 = loglog(1:length(agF),agF,'m-','LineWidth',2);
title('ObjVal vs Iterations')
xlabel('Total iterations');
ylabel('Norm of Gradient');
legend('GD','Newton-CG','L-BFGS','AGD','Location','southeast');
hold off

%Plots
figure('Name','Accuracy vs Iterations');
plot3 = semilogx(1:length(t_gdlab),t_gdlab,'b-','LineWidth',2);
hold on
plot3 = semilogx(1:length(t_ncglab),t_ncglab,'r-','LineWidth',2);
plot3 = semilogx(1:length(t_lblab),t_lblab,'g-','LineWidth',2);
plot3 = semilogx(1:length(t_aglab),t_aglab,'m-','LineWidth',2);
title('Accuracy vs Iterations')
xlabel('Total iterations');
ylabel('Correct assignments (%)');
legend('GD','Newton-CG','L-BFGS','AGD','Location','southeast');
hold off

figure('Name','ObjVal vs Time');
plot1 = loglog(gdt,gdF,'b-','LineWidth',2);
hold on
plot1 = loglog(ncgt,ncgF,'r-','LineWidth',2);
plot1 = loglog(lbt,lbF,'g-','LineWidth',2);
plot1 = loglog(agt,agF,'m-','LineWidth',2);
title('ObjVal vs Time')
xlabel('Time');
ylabel('ObjVal');
legend('GD','Newton-CG','L-BFGS','AGD','Location','southeast');
hold off

figure('Name','||G|| vs Time');
plot5 = loglog(gdt,gdG,'b-','LineWidth',2);
hold on
plot5 = loglog(ncgt,ncgG,'r-','LineWidth',2);
plot5 = loglog(lbt,lbG,'g-','LineWidth',2);
plot5 = loglog(agt,agG,'m-','LineWidth',2);
title('||G|| vs Time')
xlabel('Time');
ylabel('Norm of Gradient');
legend('GD','Newton-CG','L-BFGS','AGD','Location','southeast');
hold off

figure('Name','Accuracy vs Time');
plot5 = semilogx(t_gdt,t_gdlab,'b-','LineWidth',2);
hold on
plot5 = semilogx(t_ncgt,t_ncglab,'r-','LineWidth',2);
plot5 = semilogx(t_lbt,t_lblab,'g-','LineWidth',2);
plot5 = semilogx(t_agt,t_aglab,'m-','LineWidth',2);
title('Accuracy vs Time')
xlabel('Time');
ylabel('Correct assignments (%)');
legend('GD','Newton-CG','L-BFGS','AGD','Location','southeast');
hold off
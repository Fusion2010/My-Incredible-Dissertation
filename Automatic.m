%% Hyper-parameters Setting
% Parameters
base = 4; % the power for dimension
m = 2 ^ base / 2; % #Columns of matrix \tilde C
n = 2 ^ base; % #Rows of matrix \tilde C
k = m / 2; % #Non-zero entries in the optimal solution
mu = 0.5; % Parameter in ADMM formulation
theta = pi / 10; % Rotation Angle
gamma = 100; % Parameter in solution generator
opt_gen = 1; % Generator choice parameter: see Is_gen_underdetermined.m
% Set the condition of matrix \tilde C: please see diag_generator.m
[S_B, S_N] = diag_generator(m, 10^(1), 0, 10^(1), 0); 
% Apply the IGen2 instance generator
[d_neg, C, xopt, fvalOpt] = ls_gen_underdetermined(n, m, k, 0.5, S_B, S_N, theta, gamma, opt_gen, 2); 
d = sparse(- d_neg); 
xopt = sparse(xopt); 
% The optimal objective value by IGen2 generator
fopt = fvalOpt; 

% Display the #non-zero entries of \tilde C
display(nnz(C))

clear d_neg

%% Perform ADMM

% Initial point in ADMM
r_init = sparse(m, 1); 
omega_init = speye(n, 1); 
y_init = sparse(m, 1); 
prod = C' * C; 
tau = 0.95 / max(prod(:)); % Hyper-parameter: step size
epsilon = 0.5; % Regularization parameter
criteria = 1e-9; % Torlerance
max_time = 1e+8; % Maximal iteration time
% It should be set as false if one use Plotting Session in this code
plotting = false; 

% ADMM
[~, omega, ~, times_admm, ob_list] = ADMM(C, r_init, omega_init, ...
    y_init, d, mu, tau, epsilon, criteria, max_time, plotting, fopt); 
%% Perform IPPMM

% Convert into IPPMM formulation
[c, Q, A, b] = QP_convertor(m, n, C, d, epsilon); 
% The free variables in Theorem 2.3: this should be renmined unchanged
free_variables = 1:m; 
% Convergence torlerance
tol = criteria; 
% Maximal iteration time
maxit = max_time; 
pc = false; % Set false and do not change this
printlevel = 1; % print the primal & dual variable
% Remain unchanged 
m_eva = m; 
n_eva = n; 
fopt = fvalOpt; 
% It should be set as false if one use Plotting Session in this code
plotting = false; 

% Perform IPPMM
[~, ~, ~, ~, ~, f_list, times_ipm] = IP_PMM(c,A,Q,b,free_variables,tol,maxit,pc,printlevel, ...
    m_eva, n_eva, C, d, epsilon, fopt, plotting); 


%% Plotting Session

if max(times_ipm) > max(times_admm)
    t_horizon = times_ipm; 
else
    t_horizon = times_admm; 
end

figure;
plot(times_admm, log(ob_list), 'blue', 'LineWidth', 2); 
hold on; 
plot(times_ipm, log(f_list), 'green', 'LineWidth', 2); 
plot(t_horizon, log(f_list(end)) * ones(size(t_horizon)), 'r--', 'LineWidth', 2); % Corrected line
title('ADMM V.S IPPMM');        
xlabel('Wall Clock Time (Seconds)');
ylabel('Objective Value');
legend('Objective Value (ADMM)', 'Objective Value (IPPMM)', 'fopt', 'Location', 'Best');
legendFontSize = 16;

% Display the iteration time of each algorithm
display(max(times_admm))
display(max(times_ipm))

% Display the gap between optimal solution obtained by ADMM and IPPMM
display(log(f_list(end)) - log(ob_list(end)))
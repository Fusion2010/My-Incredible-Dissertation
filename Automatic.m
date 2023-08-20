%% 
% Parameters
base = 4; 
m = 2 ^ base / 2; 
n = 2 ^ base; 
k = m / 2; 
mu = 0.5; 
theta = pi / 10; 
gamma = 100; 
opt_gen = 1; 
[S_B, S_N] = diag_generator(m, 10^(5), 0, 10^(5), 0); 
[d_neg, C, xopt, fvalOpt] = ls_gen_underdetermined(n, m, k, 0.5, S_B, S_N, theta, gamma, opt_gen, 2); 
d = sparse(- d_neg); 
xopt = sparse(xopt); 

display(nnz(C))

clear d_neg

%% 
% Perform ADMM

r_init = sparse(m, 1); 
omega_init = sparse(n, 1); 
y_init = sparse(m, 1); 
prod = C' * C; 
tau = 0.95 / max(prod(:)); 
epsilon = 0.5; 
criteria = 1e-9; 
max_time = 1e+8; 
plotting = false; 
fopt = fvalOpt; 

[~, omega, ~, times_admm, ob_list] = ADMM(C, r_init, omega_init, ...
    y_init, d, mu, tau, epsilon, criteria, max_time, plotting, fopt); 
%%
% Perform IPPMM

[c, Q, A, b] = QP_convertor(m, n, C, d, epsilon); 
free_variables = 1:m; 
tol = criteria; 
maxit = max_time; 
pc = false; 
printlevel = 1; 
m_eva = m; 
n_eva = n; 
x_eva = xopt; 
fopt = fvalOpt; 
plotting = false; 

[~, ~, ~, ~, ~, f_list, times_ipm] = IP_PMM(c,A,Q,b,free_variables,tol,maxit,pc,printlevel, ...
    m_eva, n_eva, x_eva, C, d, epsilon, fopt, plotting); 


%% 
% Plotting Session

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

display(max(times_admm))
display(max(times_ipm))

display(log(f_list(end)) - log(ob_list(end)))
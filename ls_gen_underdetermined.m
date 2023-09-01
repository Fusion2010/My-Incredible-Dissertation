function [b, A, xopt, fvalOpt] = ls_gen_underdetermined(n, m, k, tau, S_B, S_N, theta, gamma, opt_gen, sparse_control)
% This script generates data (A,b,xopt) such that:
%                xopt := argmin tau*||x||_1 + 0.5||Ax-b||_2^2.
%
% The instance generator which is used is based on the generator IGen2 in 
% Subsection 3.2 in [1].
%
% [1] "K. Fountoulakis and J. Gondzio. Performance of First- and Second-
%      Order Methods for Big Data Optimization. Technical Report ERGO 15-005."
%
% Matrix is split into two parts, i.e., A := [B,N], where B is an m x m 
% matrix and N is a (n-m) x m matrix. Both matrices are constructed using 
% Givens rotations, see Section 4 in [1] for details. 
% 
% The optimal solution xopt is generated using procedure OsGen3 as 
% described in Section 6.0 in [1].
%
% Input:
%
%   n: number of unknown variables. It must be a power of 2
%   m: number of rows in matrix A. It must be a power of 2 and < n
%   k: number of nonzeros in the optimal solution. It must be a power of 
%      2 and <= m
%   tau: the parameter that scales the l1-norm in the objective function
%   S_B: the singular values of matrix B in A = [B N]. It must be a column 
%      vector of length m with positive entries
%   S_N: the singular values of matrix N in A = [B N]. It must be a column 
%      vector of length m with positive entries
%   theta: the rotation angle of Givens rotation. For details see Section 4 
%      of [1]
%   gamma: a parameter that controls the length of the optimal solution, 
%      see procedure OsGen3 in Section 6.0 in [1]
%   opt_gen: Choose the optimal solution generator: If opt_gen == 0, then
%   OsGen3 will be applied; Else, OsGen1 will be used
%   sparse_control: This is an indicator that how many Givens Rotation
%   Matrces will be multiplied; By default, only one will be used
%
% Output: 
%
%   b: right hand side 
%   A: m x n overdetermined matrix with rank n
%   xopt: the minimizer of tau*||x||_1 + 0.5||Ax-b||_2^2
%   fvalOpt: the value tau*||xopt||_1 + 0.5||Axopt-b||_2^2
%
% Memory requirements: Generally, we have noticed that the memory 
% requirements increase linearly with respect to n. 
% For example, if n=2^14 and m=2*n, then 3 MBs are required approximately. 
% If n=2^20 and m=n/2^3, then 190 MB are required approximately.

%% Create a composition of Givens rotations: Sparsity Control

% Create Base
Gt_B = speye(m, m); 
l_G_B = speye(m, m); 
Gt_N = speye(n - m, n - m); 
l_G_N = speye(n - m, n - m);

% The First Rotation Matrix
ct = cos(theta);
st = sin(theta);
R = [
     ct  -st
     st   ct
    ];
Rt = R';

Gt_B_1 = kron(sparse(1:m/2,1:m/2,ones(m/2,1),m/2,m/2),sparse(Rt));
l_G_B_1 = kron(sparse(1:m/2,1:m/2,ones(m/2,1),m/2,m/2),sparse(R));
Gt_N_1 = kron(sparse(1:(n-m)/2,1:(n-m)/2,ones((n-m)/2,1),(n-m)/2,(n-m)/2),sparse(Rt));
l_G_N_1 = kron(sparse(1:m/2,1:m/2,ones(m/2,1),m/2,m/2),sparse(R));

% The Second Rotation Matrix
last = sparse(1, m); 
last(end) = 1; 


Gt_B_tilde = [speye(1, m);
    sparse(m - 2, 1),  kron(sparse(1:m/2 - 1,1:m/2 - 1, ones(m/2 - 1,1),m/2 - 1,m/2 - 1),sparse(Rt)), sparse(m - 2, 1);
    last]; 
l_G_B_tilde = [speye(1, m); 
    sparse(m - 2, 1),  kron(sparse(1:m/2 - 1,1:m/2 - 1,ones(m/2 - 1,1),m/2 - 1,m/2 - 1),sparse(R)), sparse(m - 2, 1);
    last]; 
Gt_N_tilde = [speye(1, m); 
    sparse(m - 2, 1), kron(sparse(1:(n-m)/2 - 1,1:(n-m)/2 - 1,ones((n-m)/2 - 1,1),(n-m)/2 - 1,(n-m)/2 - 1),sparse(Rt)), sparse(m - 2, 1);
    last]; 

l_G_N_tilde = [speye(1, m); 
    sparse(m - 2, 1), kron(sparse(1:m/2- 1,1:m/2- 1,ones(m/2- 1,1),m/2- 1,m/2- 1),sparse(R)) , sparse(m - 2, 1);
    last]; 

for i = 1:sparse_control
    
    if rem(i, 2) == 0 
        B_left = l_G_B_tilde; 
        B_right = Gt_B_tilde; 
        N_left = l_G_N_tilde; 
        N_right = Gt_N_tilde; 
    else
        B_left = l_G_B_1; 
        B_right = Gt_B_1; 
        N_left = l_G_N_1; 
        N_right = Gt_N_1;
    end
    
    Gt_B = B_right * Gt_B; 
    l_G_B = l_G_B * B_left; 
    Gt_N = N_right * Gt_N; 
    l_G_N = l_G_N * N_left; 
end

%% Create the optimal solution xopt.
% We relax the cardinality constraint in Procedure OsGen3 in Section 6 in 
% [1]. In particular, we first solve the unconstrained problem:
%   xopt_temp := argmin_y ||G_B^Ty-gamma(Sigma_B^T*Sigma_B)^(-1)1_m||_2^2
% and then we project xopt_temp to the feasible set defined by the 
% cardinality constraints ||y||_0 <= k. We do this in order to 
% inexpensively obtain an approximate solution to the original problem in 
% Procedure OsGen3. Finally, we set xopt(1:m) = proj_xopt_temp, where 
% proj_xopt_temp is the projected temp in the feasible set 
% { y in R^m | ||y||_0 <= k }. The rest xopt(m+1:end) = 0.

% xopt_temp := argmin_y ||G_B^Ty-gamma(Sigma_B^T*Sigma_B)^(-1)1_m||_2^2
if (opt_gen == 0)
    xopt_temp = Gt_B'*(gamma./(S_B.^2)); 
    
    [~, idx] = sort(abs(xopt_temp));
    
    xopt = zeros(n,1);
    
    % Keep the k/2 smallest 
    xopt(idx(1:k/2)) = xopt_temp(idx(1:k/2));
    
    % Keep the k/2 largest
    xopt(idx(end:-1:end-k/2 + 1)) = xopt_temp(idx(end:-1:end-k/2 + 1));
% Otherwise, apply the OsGen1 Generator
else
    x_star = zeros(n, 1);
    
    for i = 1:k
        x_star(i) = (2 * gamma) * (rand() - 0.5); % Generate a random value in the range [-γ, γ]
    end
    
    xopt = x_star; 

end

%% Create noise.
if (opt_gen == 0)
    % Choose a subgradient of the l1-norm at point xopt
    sub_g = rand(m,1);
    sub_g(idx(1:k)) = sign(xopt(idx(1:k)));

    B = sparse(1:m,1:m,S_B,m,m); 
    % Create permutation matrix for B
    P_B = speye( m );
    P_B = P_B(randperm(m),:);
    B = P_B*l_G_B*P_B*sparse(1:m,1:m,S_B,m,m)*Gt_B;
    
    
    %e = tau*((Gt_B*sub_g)./S_B);
    e = P_B*l_G_B*P_B*sparse(1:m,1:m,1./S_B,m,m)*Gt_B*sub_g;
    e = tau*e;
else
    % Choose a subgradient of the l1-norm at point xopt
    sub_g = zeros(m, 1);
    for i = 1:m
        if xopt(i) == 0
            % For promising well-conditioned solution
            sub_g(i) = 0.875 + (0.125 * rand()); 
        elseif xopt(i) > 0
            sub_g(i) = 1; 
        else
            sub_g(i) = -1; 
        end
    end
    
    B = sparse(1:m,1:m,S_B,m,m); 

    % Create permutation matrix for B
    P_B = speye( m );
    P_B = P_B(randperm(m),:);
    B = P_B*l_G_B*P_B*sparse(1:m,1:m,S_B,m,m)*Gt_B;

    %e = tau*((Gt_B*sub_g)./S_B);
    e = P_B*l_G_B*P_B*sparse(1:m,1:m,1./S_B,m,m)*Gt_B*sub_g;
    e = tau*e;
end



%% Create matrix A.
if (opt_gen == 0)
    % Create permutation matrix for B
    P_N = speye( n - m );
    P_N = P_N(randperm(n - m),:);
    
    N = l_G_N*[sparse(1:m,1:m,S_N,m,m),sparse(m,n-2*m)]*(P_N*Gt_N*P_N);
    temp = abs(N'*e);
    tauNte = zeros(n-m,1);
    idx = temp ~= 0;
    l_idx = sum(idx);
    tauNte(idx) = (rand(l_idx,1).*tau)./temp(idx);
    N(:,idx) = N(:,idx)*sparse(1:l_idx,1:l_idx,tauNte(idx),l_idx,l_idx);
else
    P_N = speye( n - m );
    P_N = P_N(randperm(n - m),:);
    
    N = l_G_N*[sparse(1:m,1:m,S_N,m,m),sparse(m,n-2*m)]*(P_N*Gt_N*P_N);
    temp = abs(N'*e);
    temp_max = max(temp); 
    tauNte = zeros(n-m,1);
    idx = temp ~= 0;
    l_idx = sum(idx);
    tauNte(idx) = (rand(l_idx,1).*tau)./temp_max;
    N(:,idx) = N(:,idx)*sparse(1:l_idx,1:l_idx,tauNte(idx),l_idx,l_idx);

end

A = [B,N];
clear N;

%% Create b
b = A*xopt + e;

%% Calculate optimal objective function value
fvalOpt = tau*norm(xopt,1) + 0.5*norm(e)^2;

end

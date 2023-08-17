function [b, A, xopt, fvalOpt] = ls_gen_overdetermined(n,m,k,tau,S,theta,gamma)
% This script generates data (A,b,xopt) such that:
%                xopt := argmin tau*||x||_1 + 0.5||Ax-b||_2^2.
%
% The instance generator which is used is based on the generator IGen in 
% Subsection 3.2 in [1].
%
% [1] "K. Fountoulakis and J. Gondzio. Performance of First- and Second-
%      Order Methods for Big Data Optimization. Technical Report ERGO 15-005."
%
% Matrix A is generated using Givens rotations, see Section 4 in [1] for 
% details. 
% 
% The optimal solution xopt is generated using procedure OsGen3 as 
% described in Section 6.0 in [1].
%
% Input:
%
%   n: number of unknown variables. It must be a power of 2
%   m: number of rows in matrix A. It must be a power of 2 and >= n
%   k: number of nonzeros in the optimal solution. It must be a power of 
%      2 and <= n
%   tau: the parameter that scales the l1-norm in the objective function
%   S: the singular values of matrix A. It must be a column vector of 
%      length n with positive entries
%   theta: the rotation angle of Givens rotation. For details see Section 4 
%      of [1]
%   gamma: a parameter that controls the length of the optimal solution, 
%      see procedure OsGen3 in Section 6.0 in [1]
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
% For example, if n=2^14 and m=2*n, then 4 MBs are required approximately. 
% If n=2^20 and m=n/2^3, then 250 MB are required approximately.

%% Givens rotation matrix.
ct = cos(theta);
st = sin(theta);
R = [
     ct  -st
     st   ct
    ];
Rt = R';

%% Create right singular vectors
Gt = kron(sparse(1:n/2,1:n/2,ones(n/2,1),n/2,n/2),sparse(Rt));

%% Create left composition of Givens rotations
l_G = kron(sparse(1:m/2,1:m/2,ones(m/2,1),m/2,m/2),sparse(R));

%% Create permutation matrix
P = speye( m );
idx = randperm(m);
P = P(idx,:);

%% Create matrix A.
A = (P*l_G*P)*[sparse(1:n,1:n,S,n,n)*Gt;sparse(m-n,n)];

%% Create the optimal solution xopt.
% xopt_temp := argmin_y ||G^Ty-gamma(Sigma^T*Sigma)^(-1)1_n||_2^2
xopt_temp = Gt'*(gamma./(S.^2)); 

[~, idx] = sort(abs(xopt_temp));

xopt = zeros(n,1);

% Keep the k/2 smallest 
xopt(idx(1:k/2)) = xopt_temp(idx(1:k/2));

% Keep the k/2 largest
xopt(idx(end:-1:end-k/2 + 1)) = xopt_temp(idx(end:-1:end-k/2 + 1));

%% Create noise.
% Choose a subgradient of the l1-norm at point xopt
sub_g = zeros(n,1);
sub_g(xopt ~= 0) = sign(xopt(xopt ~= 0));
sub_g(xopt == 0) = 1.8.*rand(sum(xopt == 0),1) - 0.9;

%e_n = tau*((Gt*sub_g)./S);
%e_n_to_m = rand(m-n,1);
%e = [e_n;e_n_to_m];

e = A*Gt'*((Gt*sub_g)./(S.^2));
e = tau*e;

%% Create b
b = A*xopt + e;

%% Calculate optimal objective function value
fvalOpt = tau*norm(xopt,1) + 0.5*norm(e)^2;

end

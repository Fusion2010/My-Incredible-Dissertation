function [u, omg, f_opt] = Displayer(m, n, x_eva, C, d, epsilon)
% Displayer - Extract the optimal solution from IPPMM. 
% This function takes the optimal solution obtained by IPPMM and convert to 
% u^* omga^* and optimal function value defined in Theorem 2.3.
%
% Input:
%   m: Number of elements in vector u
%   n: Number of elements in vectors omg_positive and omg_negative
%   x_eva: Optimized variable vector [u; omg_positive; omg_negative]
%   C: Constraint matrix for the problem
%   d: Offset vector for the quadratic term
%   epsilon: Regularization parameter for the l1-norm term
%
% Output:
%   u: Extracted vector u from x_eva
%   omg: Extracted vector omg from x_eva (omg = omg_positive - omg_negative)
%   f_opt: Value of the objective function at the optimal solution
%
% If the constraint norm(u - C * omg, 2) is approximately zero, a message
% is printed indicating that the solution satisfies the constraints.
    u = x_eva(1: m); 
    omg_positive = x_eva(m + 1: m + n); 
    omg_negative = x_eva(m + n + 1: m + 2 * n); 
    omg = omg_positive - omg_negative; 
    f_opt = norm(u + d)^2 / 2 + norm(omg, 1) * epsilon;

    if (norm(u - C * omg, 2) < 1e-6)
        fprintf('Solution found by QP solver satisfies the constraints')
    end
    
end


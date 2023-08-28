function [c, Q, A, b] = QP_convertor(m, n, C, d, epsilon)
% QP_convertor - Convert optimization problem into QP standard form matrices.
% This function prepares matrices required to formulate the IPPMM 
% based on ADMM formualation. 
%
% Input:
%   m: Number of rows in constraint matrix C
%   n: Number of columns in constraint matrix C
%   C: Constraint matrix for the problem
%   d: Offset vector for the quadratic term
%   epsilon: Regularization parameter for the l1-norm term
%
% Output:
%   c: Coefficients vector for the linear term of the QP objective
%   Q: Coefficient matrix for the quadratic term of the QP objective
%   A: Coefficient matrix for the inequality constraints
%   b: Right-hand side vector for the inequality constraints

    % Precompute frequently used matrices
    eye_m = speye(m);
    zeros_m_n = sparse(m, n);
    zeros_2n_2nm = sparse(2 * n, m + 2 * n);

    % Prepare c
    c = [d; epsilon * ones(2 * n, 1)];

    % Prepare Q
    Q = [eye_m, zeros_m_n, zeros_m_n; zeros_2n_2nm];

    % Prepare A
    A = [-eye_m, C, -C];

    % Prepare b
    b = zeros(m, 1);
end

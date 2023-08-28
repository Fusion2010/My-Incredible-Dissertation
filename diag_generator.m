function [S_B, S_N] = diag_generator(m, eig_max_S, eig_min_S, eig_max_N, eig_min_N)
% diag_generator - Generate diagonal matrices S_B and S_N for singular values, \
% which will be required by IP_PMM.m
% This function generates random diagonal matrices S_B and S_N with
% singular values in specified ranges.
%
% Input:
%   m: Dimension of the diagonal matrices
%   eig_max_S: Maximum eigenvalue for matrix S_B
%   eig_min_S: Minimum eigenvalue for matrix S_B
%   eig_max_N: Maximum eigenvalue for matrix S_N
%   eig_min_N: Minimum eigenvalue for matrix S_N
%
% Output:
%   S_B: Diagonal matrix S_B with random eigenvalues
%   S_N: Diagonal matrix S_N with random eigenvalues
    S_B = eig_min_S + (eig_max_S - eig_min_S) * rand(m, 1); 
    S_N = eig_min_N + (eig_max_N - eig_min_N) * rand(m, 1); 
end
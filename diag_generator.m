function [S_B, S_N] = diag_generator(m, eig_max_S, eig_min_S, eig_max_N, eig_min_N)
    S_B = eig_min_S + (eig_max_S - eig_min_S) * rand(m, 1); 
    S_N = eig_min_N + (eig_max_N - eig_min_N) * rand(m, 1); 
end
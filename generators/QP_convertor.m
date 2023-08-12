function [c, Q, A, b] = QP_convertor(m, n, C, d, epsilon)
    c = [d; epsilon * ones(n, 1); epsilon * ones(n, 1)]; 
    Q = sparse([eye(m, m), zeros(m, 2 * n); zeros(2 * n, m + 2 * n)]); 
    A = sparse([-eye(m, m), C, -C]); 
    b = zeros(m, 1); 
end
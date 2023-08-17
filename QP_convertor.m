function [c, Q, A, b] = QP_convertor(m, n, C, d, epsilon)
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

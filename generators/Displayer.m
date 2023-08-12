function [u, omg, f_opt] = Displayer(m, n, x_eva, C, d, epsilon)

    u = x_eva(1: m); 
    omg_positive = x_eva(m + 1: m + n); 
    omg_negative = x_eva(m + n + 1: m + 2 * n); 
    omg = omg_positive - omg_negative; 
    f_opt = norm(u + d)^2 / 2 + norm(omg, 1) * epsilon;

    if (norm(u - C * omg, 2) < 1e-6)
        fprintf('Solution found by QP solver satisfies the constraints')
    end
    
end


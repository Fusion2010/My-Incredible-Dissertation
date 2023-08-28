function [r, omega, y, elapsed_times, ob_list] = ADMM(C, r_init, omega_init, ...
    y_init, d, mu, tau, epsilon, criteria, max_time, plotting, fopt)
% ADMM - Alternating Direction Method of Multipliers solver.
% This function solves a constrained optimization problem using the
% Alternating Direction Method of Multipliers (ADMM) algorithm.
%
% Inputs:
%   C: Constraint matrix
%   r_init: Initial guess for variable r
%   omega_init: Initial guess for variable omega
%   y_init: Initial guess for variable y
%   d: Offset vector
%   mu: ADMM parameter for augmented Lagrangian term
%   tau: ADMM parameter for primal update
%   epsilon: Regularization parameter for soft-thresholding
%   criteria: Convergence criteria (minimum dual/primal gap)
%   max_time: Maximum number of iterations
%   plotting: Boolean indicating whether to plot convergence curve
%   fopt: Optimal objective value (for plotting comparison)
%
% Outputs:
%   r: Solution for variable r
%   omega: Solution for variable omega
%   y: Solution for variable y
%   elapsed_times: Elapsed time at each iteration
%   ob_list: Objective value at each iteration

    % Define nested functions for sub-steps of the ADMM algorithm
    % Soft Threshold in ADMM
    function result = soft(x, threshold)
        % Soft-thresholding operator
        result = sign(x) .* max(abs(x) - threshold, 0);
    end
    
    % Projection Operator in ADMM
    function result = projection_operator(x, epsilon)
        % Projection operator onto a ball
        if norm(x, 2) <= epsilon
            result = x;
        else
            result = epsilon * x / norm(x, 2);
        end
    end
    
    % Update in auxiliary variable r
    function result = r_update(C, omega, d, y, mu, epsilon)
        % Update r variable
        temp_r = C * omega + d - y / mu;
        result = projection_operator(temp_r, epsilon);
    end
    
    % Update in primal variable \omega
    function result = omega_update(C, omega, d, r, y, mu, tau)
        % Update omega variable
        grad = C' * (C * omega + d - r - y / mu);
        temp_omg = omega - tau * grad;
        result = soft(temp_omg, tau / mu);
    end
    
    % Update in dual variable y
    function [dual_gap, result] = y_update(C, omega, d, r, y, mu)
        % Update y variable
        y_temp = C * omega + d - r;
        dual_gap = norm(y_temp, 2);
        result = y - mu * y_temp;
    end

    % Initialize variables and counters
    r = r_init;
    omega = omega_init;
    y = y_init;

    it = 0; % Initialize the iteration index
    min_gap = 1e+6;  % Initialize to a high value
    
    % Preallocate memory for performance improvement
    elapsed_times = zeros(1, max_time);
    ob_list = zeros(1, max_time);
    
    tic;  % Start CPU timer
    
    % Main ADMM loop
    while (it < max_time) && (min_gap > criteria) 
        prev_omega = omega; 
        
        % Update variables
        r = r_update(C, omega, d, y, mu, epsilon);
        omega = omega_update(C, omega, d, r, y, mu, tau);
        [dual_gap, y] = y_update(C, omega, d, r, y, mu);
        
        % Calculate primal gap
        primal_gap = norm(omega - prev_omega, 2);  
        min_gap = min(dual_gap, primal_gap);
        
        it = it + 1; % update iteration index
        
        % Display minimal gap between primal & dual
        fprintf('Minimum of dual and primal gap at iteration %d is %f\n', it, min_gap);
        
        % Store objective values for plotting
        if (~plotting)
            obj1 = norm(C * omega + d, 2)^2 / 2;
            obj2 = norm(omega, 1) * epsilon;
            elapsed_times(it) = toc;            
            ob_list(it) = obj1 + obj2;
        end
    end

    % Trim unused memory
    elapsed_times = elapsed_times(1: it); 
    ob_list = ob_list(1: it); 

    fprintf('%d iterations in total to converge under criteria %f\n', it, criteria);
    
    % Plot if requested
    if plotting
        figure;
        plot(elapsed_times, ob_list); 
        hold on; 
        plot(elapsed_times, fopt * ones(size(elapsed_times)), "r--", 'LineWidth', 2)
        title('Alternating Direction Method of Multipliers');        
        xlabel('Iteration');
        ylabel('Objective Value');
        legend('Objective Value', 'fopt', 'Location', 'Best');
        hold off;
    end
end
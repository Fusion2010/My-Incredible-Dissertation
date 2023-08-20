function [r, omega, y, elapsed_times, ob_list] = ADMM(C, r_init, omega_init, ...
    y_init, d, mu, tau, epsilon, criteria, max_time, plotting, fopt)
    
    function result = soft(x, threshold)
        result = sign(x) .* max(abs(x) - threshold, 0);
    end

    function result = projection_operator(x, epsilon)
        if norm(x, 2) <= epsilon
            result = x;
        else
            result = epsilon * x / norm(x, 2);
        end
    end

    function result = r_update(C, omega, d, y, mu, epsilon)
        temp_r = C * omega + d - y / mu;
        result = projection_operator(temp_r, epsilon);
    end

    function result = omega_update(C, omega, d, r, y, mu, tau)
        grad = C' * (C * omega + d - r - y / mu);
        temp_omg = omega - tau * grad;
        result = soft(temp_omg, tau / mu);
    end

    function [dual_gap, result] = y_update(C, omega, d, r, y, mu)
        y_temp = C * omega + d - r;
        dual_gap = norm(y_temp, 2);
        result = y - mu * y_temp;
    end


    r = r_init;
    omega = omega_init;
    y = y_init;
    it = 0;
    min_gap = 1e+6;  
    elapsed_times = zeros(1, max_time); % Preallocate memory
    ob_list = zeros(1, max_time); % Preallocate memory

    % if 1 / tau < max(eig(C' * C))
    %     error('Error: Convergence Theorem is not met!!!!!!! Tau should be less than Max-Eigenvalue');
    % end
    
    tic; 
    while (it < max_time) && (min_gap > criteria) 
        
        % start_time = cputime; % Start GPU timer
        
        prev_omega = omega; 
        r = r_update(C, omega, d, y, mu, epsilon);
        omega = omega_update(C, omega, d, r, y, mu, tau);
        
        [dual_gap, y] = y_update(C, omega, d, r, y, mu);
        primal_gap = norm(omega - prev_omega, 2);  
        min_gap = min(dual_gap, primal_gap);
        
        it = it + 1;

        fprintf('Minimun of dual and primal gap at %d iteration is %f\n', it, min_gap);
        
        if (~plotting)
            obj1 = norm(C * omega + d, 2)^2/2;
            obj2 = norm(omega, 1) * epsilon;
            elapsed_times(it) = toc;            
            ob_list(it) = obj1 + obj2;
        end
    end

    elapsed_times = elapsed_times(1: it); 
    ob_list = ob_list(1:it); 

    fprintf('%d iterations in total to converge under criteria %f\n', it, criteria);
    
    if plotting
        figure;
        plot(elapsed_times, ob_list); 
        hold on; 
        plot(elapsed_times, fopt * ones(size(elapsed_times)), "r--", 'LineWidth', 2)
        title('Alternating Dieration Method of Multipliers');        
        xlabel('Iteration');
        ylabel('Objective Value');
        legend('Objective Value', 'fopt', 'Location', 'Best');
        hold off;
    end
    
    
end

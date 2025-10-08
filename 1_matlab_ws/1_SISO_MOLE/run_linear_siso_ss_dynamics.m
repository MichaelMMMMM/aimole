function [y, X] = run_linear_siso_ss_dynamics(A, B, C, x0, u)
    % Number samples
    N = length(u);
    % Allocate
    y = zeros(N,1);
    X = zeros(length(x0), N+1);
    % Init
    x = x0;
    % Itearte dynamics
    for n = 1:N
        % Save current state
        X(:, n) = x;
        % Update state
        x = A*x + B*u(n);
        % Save output
        y(n) = C*x;
    end
    % Save final state sample
    X(:, end) = x;
end


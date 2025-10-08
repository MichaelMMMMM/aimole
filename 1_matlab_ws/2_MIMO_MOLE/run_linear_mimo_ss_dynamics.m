function [y, X] = run_linear_mimo_ss_dynamics(A, B, C, x0, u)
    % Number samples
    R = size(B, 2);
    M = size(C, 1);
    N = length(u)/R;
    % Allocate
    y = zeros(N*M,1);
    X = zeros(length(x0), N+1);
    % Init
    x = x0;
    % Itearte dynamics
    for n = 1:N
        % Save current state
        X(:, n) = x;
        % Update state
        x = A*x + B*u(1+R*(n-1):R*n, 1);
        % Save output
        y(1+M*(n-1):M*n, 1) = C*x;
    end
    % Save final state sample
    X(:, end) = x;
end


function [y, X, t] = run_double_pendulum_feedforward_dynamics(u)
%RUN_DOUBLE_PENDULUM_FEEDFORWARD_DYNAMICS The function simulates the
%feedforward dynamics of a double-pendulum.
%   The double-pendulum starts from its down-wards, asymptotically stable
%   equilibrium. The system's input variable is a motor torque driving the
%   first pendulum. The system's output variable is the absolute angle of
%   the second pendulum. The input variable u of the function is a
%   feedforward input trajectory of length N that is applied to the system
%   dynamics. The first output variable y of the function is the system's output
%   trajectory of length N that is recorded with a phase lead of m=1. 
%   The second output variable X of the function is a 4xN matrix containing
%   the state trajectory.
%   the third output variable t of the function is the time vector
%   corresponding to the sampling times of the output trajectory.
    % Trajectory Length
    N = length(u);
    % Initial state
    x = zeros(4,1);
    % Allocate output trajectory
    y = zeros(N,1);
    % Allocate state trajectory
    X = zeros(4, N+1);
    X(:, 1) = x;
    % Sampling period
    T = 0.05;
    % Iteratively compute the dynamics
    for n = 1:N
        % Input sample
        un = u(n,1);
        % Integrate nonlinear dynamics
        ode_function = @(x, t)dx_double_pendulum(x, t, un);
        [~, Xode]       = ode45(ode_function, [0, T], x);
        % Extract updated state
        x = Xode(end, :)';
        % Compute the absolute angle of the second pendulum
        yn = x(1,1) + x(3,1);
        % Save the output sample
        y(n,1)  = yn;
        % Save the state sample
        X(:, n+1) = x;
    end
    % Setup the time vector
    t = T*(0:N-1)';
end


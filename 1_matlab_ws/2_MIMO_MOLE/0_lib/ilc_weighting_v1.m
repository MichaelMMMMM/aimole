function [W, S] = ilc_weighting_v1(P, number_samples, number_outputs, number_inputs)
    % Sizes
    N = number_samples;
    Q = number_outputs;
    R = number_inputs;
    % Determine weights
    W = eye(Q*N);
    S = norm(P)^2*eye(R*N);
end


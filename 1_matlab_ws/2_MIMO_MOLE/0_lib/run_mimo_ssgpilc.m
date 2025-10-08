function [ev, Xc, yc, uc] = run_mimo_ssgpilc(number_trials, ...
                                                    reference, ...
                                                    initial_input, ...
                                                    weighting_function, ...
                                                    number_training_trials, ...
                                                    dynamics_function, ...
                                                    number_outputs, ...
                                                    number_inputs, ...
                                                    number_states, ...
                                                    output_matrix)
    % Sizes
    Q = number_outputs;
    M = number_states;
    R = number_inputs;
    N = length(reference)/Q;
    r = reference;
    J = number_trials;
    H = number_training_trials;
    % Allocate
    ev = zeros(J,1);
    Xc = cell(J,1);
    yc = cell(J,1);
    uc = cell(J,1);
    % Init ILC
    u  = initial_input;
    for j = 1:J
        % Trial
        [y, X] = dynamics_function(u);
        yc{j,1} = y;
        Xc{j,1} = X;
        uc{j,1} = u;
        % GP-Model
        gp_model = CFullSSGP(M, R, N, H, output_matrix);
        gp_model.train_gp_model(Xc(1:j,1), uc(1:j,1));
        % ILC-Design
        P = gp_model.linearize_at_input_trajectory(zeros(M,1), u);
        [W, S] = weighting_function(P, N, Q, R);
        L = (P'*W*P+S)\P'*W;
        % ILC update
        e = r-y;
        u = u + L*e;
        ev(j,1) = norm(e);
    end
end


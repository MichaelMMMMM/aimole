classdef CMRTMOLE < handle
    %CMRTMOLE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        mole
    end
    
    methods
        function obj = CMRTMOLE(dyn_func, ...
                                number_inputs, ...
                                number_states, ...
                                output_matrix, ...
                                number_trials_for_training)
            obj.mole = CMIMOMOLE(dyn_func, ...
                                 number_inputs, ...
                                 number_states, ...
                                 output_matrix, ...
                                 number_trials_for_training);
        end
        
        function emat = run_vanilla_mole(rc, nbr_trials_per_ref, uIc)
            % Sizes
            W = length(rc);
            J = nbr_trials_per_ref;

            % Allocate return objects
            emat = zeros(J, W);
            % Iterate references
            for w = 1:W
                % Fetch reference and initial input
                r = rc{w};
                u = uIc{w};
                % Run MIMO-MOLE
                emat(:, w) = obj.mole.run_mimo_mole(r, u, J);
            end
        end
        function emat = run_mrt_mole(obj, rc, nbr_trials_per_ref, uIc)
            % Sizes
            W = length(rc);
            J = nbr_trials_per_ref;
            N = length(rc{1});

            % Allocate return objects
            emat = zeros(J, W);
            Xc   = cell(J, W);
            Uc   = cell(J, W);
            Yc   = cell(J, W);
            % Iterate references
            for w = 1:W
                % Fetch reference and initial input
                r = rc{w};
                u = uIc{w};

                % Determine initial input 
                if(w > 1)
                    % Training data for global GP model
                    XTc = reshape(Xc(:, 1:w-1), (w-1)*J, 1); 
                    UTc = reshape(Uc(:, 1:w-1), (w-1)*J, 1);

                    % Train GP
                    global_gp = CMIMOSSGP(obj.mole.number_states, ...
                                          obj.mole.number_inputs, ...
                                          N,...
                                          obj.mole.output_matrix);
                    global_gp.train_gp_model(XTc, UTc);
                    % Find optimal input
                    prediction_function    = @(u)global_gp.predict(u, zeros(obj.mole.number_states,1));
                    linearization_function = @(u)global_gp.linearize_at_input_trajectory(zeros(obj.mole.number_states,1), u);
                    cost_function          = @(u)obj.cost_input_prediction(u, r, prediction_function, linearization_function);
                    optim_opt = optimoptions('fminunc', 'SpecifyObjectiveGradient',true);
                    optim_opt.OptimalityTolerance = 1e-3;
                    optim_opt.StepTolerance       = 1e-3;
                    optim_opt.FunctionTolerance   = 1e-3;
                    optim_opt.Display             = "none";
                    u = fminunc(cost_function, zeros(length(u),1), optim_opt);
                end

                % Run MIMO-MOLE
                [emat(:, w), ~, Yc(:, w), Uc(:, w), Xc(:, w)] = obj.mole.run_mimo_mole(r, u, J);
            end
        end
        function [cost, gradient] = cost_input_prediction(~, u, r, prediction_function, linearization_function)
            % Output prediction
            y = prediction_function(u);
            % Tracking error
            e = r - y;
            % Quadratic cost
            cost = e'*e;
            % Gradient
            gradient = -2*linearization_function(u)'*e;
        end
    end
end


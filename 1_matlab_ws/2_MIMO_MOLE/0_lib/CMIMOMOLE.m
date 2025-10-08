classdef CMIMOMOLE < handle
    %CMIMOMOLE Class to implement AI-MOLE for multi-input/multi-output dynamics.
    
    properties
        dyn_func
        number_inputs
        number_states
        output_matrix
        number_trials_for_training
    end
    
    methods
        function obj = CMIMOMOLE(dyn_func, ...
                                 number_inputs, ...
                                 number_states, ...
                                 output_matrix, ...
                                 number_trials_for_training)
            obj.dyn_func      = dyn_func;
            obj.number_inputs = number_inputs;
            obj.number_states = number_states;
            obj.output_matrix = output_matrix;
            if(nargin < 5)
                obj.number_trials_for_training = 3;
            else
                obj.number_trials_for_training = number_trials_for_training;
            end
        end
        
        function [ev, ec, yc, uc] = run_mimo_mole(obj, ...
                                                  reference, ...
                                                  initial_input_trajectory, ...
                                                  number_of_trials)
            % Init
            r = reference;
            u = initial_input_trajectory;
            J = number_of_trials;
            M = obj.number_states;
            O = size(obj.output_matrix, 1);
            N = length(r)/O;
            R = obj.number_inputs;
            C = obj.output_matrix;
            gp_model = CMIMOSSGP(M, R, N, C);

            % Allocate return objects
            ev = zeros(J,1);
            ec = cell(J,1);
            Xc = cell(J,1);
            yc = cell(J,1);
            uc = cell(J,1);
            
            % Iterate
            for j = 1:J
                % Run trial
                [y, X] = obj.dyn_func(u);

                % Save input/output trajectories
                yc{j,1} = y;
                Xc{j,1} = X;
                uc{j,1} = u;

                % -- Model Learning
                % Retrieve training data
                H = obj.number_trials_for_training;
                if(j < H)
                    ut_c = uc(1:j, 1);
                    Xt_c = Xc(1:j, 1);
                else
                    ut_c = uc(j+1-H:j,1);
                    Xt_c = Xc(j+1-H:j,1);
                end
                % Train gp model
                gp_model.train_gp_model(Xt_c, ut_c);
                % Linearize gp model at current input trajectory
                P = gp_model.linearize_at_input_trajectory(zeros(M,1), u);

                % -- ILC Update
                % NO-ILC weights
                [W, S] = obj.ilc_weighting(P);
                % Learning gain matrix
                L = (P'*W*P+S)\P'*W;
                % Tracking error
                e = r - y;
                % Input update
                u = u + L*e;

                % Save error data
                ec{j,1} = e;
                ev(j,1) = norm(e);
            end
        end
        function [W, S] = ilc_weighting(obj, P)
            % Sizes
            N = size(P, 2)/obj.number_inputs;
            Q = size(obj.output_matrix, 1);
            R = obj.number_inputs;
            % Determine weights
            Pc = obj.P_2_subP(P);
            wv = zeros(Q, 1);
            for q = 1:Q
                wv(q,1) = 1/norm(cell2mat(Pc(q, :)));
            end
            W  = diag(repmat(wv, N,1));
            sv = zeros(R,1);
            for r = 1:R
                sv(r,1) = norm(cell2mat(Pc(:, r)));
            end
            S  = 0.1*diag(repmat(sv, N,1));
        end
        function Pc = P_2_subP(obj, P)
            % Sizes
            Q = size(obj.output_matrix, 1);
            R = obj.number_inputs;
            % Allocate
            Pc = cell(Q, R);
            for q = 1:Q
                for r = 1:R
                    Pc{q,r} = P(q:Q:end, r:R:end);
                end
            end
        end
    end
end


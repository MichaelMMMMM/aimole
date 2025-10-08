classdef CISMOLE < handle
    %CISMOLE Class to represent and implement the input/state version of
    %AI-MOLE for SISO dynamics. The class is implemented such that for each
    %system a respective instance of CISMOLE is instanciated using the ctor
    %and that for each respective reference tracking task run_iomole is
    %called.
    
    properties
        dyn_func
        number_states
        output_matrix
        number_trials_for_training
    end
    
    methods
        function obj = CISMOLE(dyn_func, ...
                               number_states, ...
                               output_matrix, ...
                               number_trials_for_training)
            obj.dyn_func                   = dyn_func;
            obj.number_states = number_states;
            obj.output_matrix = output_matrix;
            if(nargin < 4)
                obj.number_trials_for_training = 3;
            else
                obj.number_trials_for_training = number_trials_for_training;
            end
        end
        
        function [ev, ec, yc, uc] = run_ismole(obj, ...
                                               reference, ...
                                               initial_input_trajectory, ...
                                               number_of_trials)
            % Init
            r = reference;
            u = initial_input_trajectory;
            J = number_of_trials;
            N = length(r);
            M = obj.number_states;
            C = obj.output_matrix;
            gp_model = CSISOSSGP(M, N, C);

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
                S = 0.1*norm(P)^2*eye(N);
                % Learning gain matrix
                L = (P'*P+S)\P';
                % Tracking error
                e = r - y;
                % Input update
                u = u + L*e;

                % Save error data
                ec{j,1} = e;
                ev(j,1) = norm(e);
            end
        end
    end
end


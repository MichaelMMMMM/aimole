classdef CIOMOLE < handle
    %CIOMOLE Class to represent and implement the input/output version of
    %AI-MOLE for SISO dynamics. The class is implemented such that for each
    %system a respective instance of CIOMOLE is instanciated using the ctor
    %and that for each respective reference tracking task run_iomole is
    %called.
    
    properties
        dyn_func
        number_trials_for_training
        


    end
    
    methods
        function obj = CIOMOLE(dyn_func, ...
                               number_trials_for_training)
            obj.dyn_func                   = dyn_func;
            if(nargin < 2)
                obj.number_trials_for_training = 3;
            else
                obj.number_trials_for_training = number_trials_for_training;
            end
        end
        
        function [ev, ec, yc, uc] = run_iomole(obj, ...
                                               reference, ...
                                               initial_input_trajectory, ...
                                               number_of_trials)
            % Init
            r = reference;
            u = initial_input_trajectory;
            J = number_of_trials;
            N = length(r);
            gp_model = CSISOFIRGP(N);

            % Allocate return objects
            ev = zeros(J,1);
            ec = cell(J,1);
            yc = cell(J,1);
            uc = cell(J,1);
            
            % Iterate
            for j = 1:J
                % Run trial
                y = obj.dyn_func(u);

                % Save input/output trajectories
                yc{j,1} = y;
                uc{j,1} = u;

                % -- Model Learning
                % Retrieve training data
                H = obj.number_trials_for_training;
                if(j < H)
                    ut_c = uc(1:j, 1);
                    yt_c = yc(1:j, 1);
                else
                    ut_c = uc(j+1-H:j,1);
                    yt_c = yc(j+1-H:j,1);
                end
                % Train gp model
                gp_model.train_gp_model(yt_c, ut_c);
                % Linearize gp model at current input trajectory
                P = gp_model.linearize_at_input_trajectory(u);

                % -- ILC Update
                % NO-ILC weights
                S = norm(P)^2*eye(N);
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


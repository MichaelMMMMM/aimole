classdef CSISOFIRGP < handle
    %CSISOFIRGP Class to implement the FIR-GP model of unknown, nonlinear,
    %single-input/single-output dynamics for the input/output-version of AI-MOLE.
    
    properties
        y_cell                      % Cell containing output trajectories for training
        u_cell                      % Cell containing input trajectories for training
        number_samples_per_trial    % Number of samples in a trajectory
        VT                          % Place holder for regression traning matrix
        gp                          % GP that is to be trained
    end
    
    methods
        function obj = CSISOFIRGP(number_samples_per_trial)
            obj.y_cell = {};
            obj.u_cell = {};
            obj.number_samples_per_trial = number_samples_per_trial;
            VT         = [];
            gp         = [];
        end
        function y = predict(obj, u)
            % Regression matrix
            V = obj.input_trajectory_2_regression_matrix(u);
            % GP prediction
            y = predict(obj.gp, V');
        end
        function train_gp_model(obj, y_cell, u_cell)
            % Save training data
            obj.y_cell = y_cell;
            obj.u_cell = u_cell;
            % Regression training matrix
            obj.regression_training_matrix();
            % Observation training vector
            zT = obj.observation_training_vector();
            % Optimization options
            opt_options = statset('fitrgp');
            opt_options.TolFun = 1e-6;
            opt_options.TolX   = 1e-8;
            % GP Training
            obj.gp = fitrgp(obj.VT', zT, 'BasisFunction', 'none', 'KernelFunction', 'squaredexponential', 'OptimizerOptions', opt_options);
        end
        function P = linearize_at_input_trajectory(obj, u)
            % Sizes
            N = obj.number_samples_per_trial;
            % Allocate
            P = zeros(N, N);
            % Iteratively compute gradients of output samples
            for n = 1:N
                % Input to regression vector
                Tn = obj.input_trajectory_2_regression_vector_matrix(n);
                % Regression vector
                vn = Tn*u;
                % Linearization
                dy_dv = obj.linearize_at_regression_vector(vn);
                % Output derivative with respect to input trajectory by the  chain rule
                P(n, :) = dy_dv*Tn;
            end 
        end
        function dy_dv = linearize_at_regression_vector(obj, v)
            % Retrieve alpha vector
            a  = obj.gp.Impl.AlphaHat;
            % Kernel Function
            kfcn = obj.gp.Impl.Kernel.makeKernelAsFunctionOfXNXM(obj.gp.Impl.ThetaHat);
            % Squared inverse of length scale
            linv = 1/(obj.gp.KernelInformation.KernelParameters(1))^2;
            % Kernel matrix of regression vectov and regression training matrix VT
            K_vVT = kfcn(v', obj.VT');
            % Allocate gradient  of K_vVT with respect to v
            grad_K_vVT = zeros(length(v), size(obj.VT, 2));
            % Iteratively compute gradient
            for n = 1:size(obj.VT,2)
                grad_K_vVT(:, n) = K_vVT(1, n)*linv*(obj.VT(:, n)- v);
            end
            % Compute linearization
            dy_dv = (grad_K_vVT*a)';
        end
        function zT = observation_training_vector(obj)
            % Sizes
            N = obj.number_samples_per_trial;
            J = length(obj.u_cell);
            % Allocate
            zT = zeros(N*J,1);
            % Iteratively compute observation training vector
            for j = 1:J
                yj                  = obj.y_cell{j,1};
                zT(1+N*(j-1):N*j,1) = yj; 
            end
        end
        function regression_training_matrix(obj)
            % Sizes
            J = length(obj.u_cell);
            N = obj.number_samples_per_trial;
            % Allocate
            obj.VT = zeros(N, J*N);
            % Itertively compute regression matrices
            for j = 1:J
                obj.VT(:, 1+N*(j-1):N*j) = obj.input_trajectory_2_regression_matrix(obj.u_cell{j,1});
            end
        end
        function V = input_trajectory_2_regression_matrix(obj, u)
            % Sizes
            N = obj.number_samples_per_trial;
            % Allocate
            V = zeros(N, N);
            % Iteratively compute regression vectors
            for n = 1:N
                % Transformation matrix
                Tn      = obj.input_trajectory_2_regression_vector_matrix(n);
                V(:, n) = Tn*u;
            end
        end
        function Tn = input_trajectory_2_regression_vector_matrix(obj, sample_index)
            %TODO Simplify!
            % Sizes
            n = sample_index;
            N = obj.number_samples_per_trial;
            % Compute transformation matrix
            Tn = [fliplr(eye(n)), zeros(n, N-n); ...
                   zeros(N-n, n), zeros(N-n, N-n)];
        end
    end
end


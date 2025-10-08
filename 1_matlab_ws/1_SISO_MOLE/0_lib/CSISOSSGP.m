classdef CSISOSSGP < handle
    %CSISOSSGP Class to implement the SS-GP model of unknown, nonlinear,
    %single-input/single-output dynamics for the input/state-version of AI-MOLE.
    
    properties
        number_states
        number_samples_per_trial
        gp_cell
        X_cell
        u_cell
        output_matrix
        VT
    end
    
    methods
        function obj = CSISOSSGP(number_states, ...
                                 number_samples_per_trial,...
                                 output_matrix)
            obj.number_states = number_states;
            obj.number_samples_per_trial = number_samples_per_trial;
            obj.gp_cell = cell(0,0);
            obj.X_cell  = cell(0,0);
            obj.u_cell  = cell(0,0);
            obj.output_matrix = output_matrix;
        end
        function [y, X, y_std, y_int] = predict(obj, u, x0)
            % Sizes
            N = length(u);
            M = obj.number_states;
            % Allocate
            y = zeros(N,1);
            X = zeros(M, N+1);
            y_std = zeros(N,1);
            y_int = zeros(N,2);
            % Init
            x = x0;
            x_std = zeros(M,1);
            x_int = zeros(M,2);
            X(:, 1) = x;
            % Iteratively compute predictions
            for n = 1:N
                % Fetch input sample
                un = u(n,1);
                % Regression vector
                v  = obj.regression_vector(x, un);
                % Iteratively predict states
                for m = 1:M
                    [x(m,1), x_std(m,1), x_int(m,:)] = predict(obj.gp_cell{m,1}, v');
                end
                % Save values
                X(:, n+1) = x;
                y(n,1)    = obj.output_matrix * x;
                y_std(n,1) = obj.output_matrix * x_std;
                y_int(n,:) = obj.output_matrix*x_int;
            end
        end
        function train_gp_model(obj, X_cell, u_cell)
            % Training Data
            obj.X_cell = X_cell;
            obj.u_cell = u_cell;
            % Regression training matrix
            obj.VT = obj.regression_training_matrix();
            % Observation training vectors
            zT_cell = obj.observation_training_vector_cell();
            % Training
            for m = 1:obj.number_states
                % Optimization options
                opt_options = statset('fitrgp');
                opt_options.TolFun = 1e-2;
                opt_options.TolX   = 1e-4;
                opt_options.UseParallel = true;
                % GP Training
                obj.gp_cell{m,1} = fitrgp(obj.VT', zT_cell{m,1}, 'BasisFunction', 'none', 'KernelFunction', 'ardsquaredexponential', 'OptimizerOptions', opt_options);
            end
        end
        function [P, A_cell, B_cell] = linearize_at_input_trajectory(obj, x0, u)
            % Sizes
            N = obj.number_samples_per_trial;
            M = obj.number_states;
            % Allocate dynamic cells
            A_cell = cell(N,1);
            B_cell = cell(N,1);
            % Iteratively compute state-space linearizations
            x = x0;
            for n = 1:N
                % Current input sample
                un = u(n, 1);
                % Regression vector
                v = obj.regression_vector(x, un);
                % Linearize at current regression vector
                [A_cell{n,1}, B_cell{n,1}] = obj.linearize_at_regression_vector(v);
                % Update state
                [~, X] = obj.predict(un, x);
                x = X(:, 2);
            end
            % Compute lifted systems matrix
            % Allocate input to state matrix
            Px = zeros(N*M, N);
            % Iteratively compute state Markov parameters
            for n = 1:N
                % Retrieve current state and input matrix
                A = A_cell{n,1};
                B = B_cell{n,1};
                if(n == 1)
                    Px(1:M, 1) = B;
                else
                    % Iteratively update the matrix columns of Px
                    for p = 1:n-1
                        Px(1+M*(n-1):M*n, p) = A*Px(1+M*(n-2):M*(n-1), p);
                    end
                    % Add the B entry
                    Px(1+M*(n-1):M*n, n) = B;
                end
            end
            % Compute P by multiplication with C
            C = obj.output_matrix;
            Q = size(C,1);
            P = zeros(Q*N, N);
            for n = 1:N
                P(1+Q*(n-1):Q*n,:) = C*Px(1+M*(n-1):M*n,:);
            end
        end
        function [A, B]  = linearize_at_regression_vector(obj, v)
            % Sizes
            n = obj.number_states;
            % Allocate
            A = zeros(n, n);
            B = zeros(n, 1);
            % Iteratively compute rows of A and B
            for m = 1:n
                % Retrieve current GP
                gp = obj.gp_cell{m, 1};
                % Retrieve alpha vector
                a  = gp.Impl.AlphaHat;
                % Kernel Function
                kfcn = gp.Impl.Kernel.makeKernelAsFunctionOfXNXM(gp.Impl.ThetaHat);
                % Squared inverse of length scale
                l = gp.KernelInformation.KernelParameters(1:n+1);
                Linv = diag(l.^(-2));
                % Kernel matrix of regression vectov and regression
                % training matrix VT
                K_vVT = kfcn(v', obj.VT');
                % Allocate gradient  of K_vVT with respect to v
                grad_K_vVT = zeros(length(v), size(obj.VT, 2));
                % Iteratively compute gradient
                for t = 1:size(obj.VT,2)
                    grad_K_vVT(:, t) = K_vVT(1, t)*Linv*(obj.VT(:, t)- v);
                end
                % Compute linearization
                AB      = grad_K_vVT*a;
                A(m, :) = AB(1:obj.number_states,1)';
                B(m, :) = AB(obj.number_states+1:end,1)';
            end
        end
        function zT_cell = observation_training_vector_cell(obj)
            % Sizes
            N = obj.number_samples_per_trial;
            M = obj.number_states;
            J = length(obj.u_cell);
            % Allocate
            zT_cell = cell(M,1);
            % Iteratively compute observation training vectors
            for m = 1:M
                % Allocate
                zT = zeros(N*J,1);
                % Iiteratively compute observation training vectors
                for j = 1:J
                    Xj = obj.X_cell{j,1};
                    zT(1+N*(j-1):N*j) = Xj(m, 2:end)';
                end
                % Save observation training vector
                zT_cell{m,1} = zT;
            end
        end
        function VT = regression_training_matrix(obj)
            % Sizes
            N = obj.number_samples_per_trial;
            m = obj.number_states;
            J = length(obj.X_cell);
            % Allocate regression training matrix
            VT = zeros(m+1, N*J);
            % Iteratively compute regression matrices
            for j = 1:J
                VT(:, 1+N*(j-1):N*j) = obj.regression_matrix(obj.X_cell{j,1}, obj.u_cell{j,1});
            end
        end
        function V = regression_matrix(obj, X, u)
            % Sizes
            N = obj.number_samples_per_trial;
            m = obj.number_states;
            % Allocate regression matrix
            V = zeros(m+1, N);
            % Iteratively compute regression vectors
            for n = 1:N
                V(:, n) = obj.regression_vector(X(:,n), u(n));
            end
        end
        function v = regression_vector(obj, x, u)
            v = [x; u];
        end
    end
end


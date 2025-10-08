classdef CMIMOSSGP < handle
    %CMIMOSSGP Class to implement the SS-GP model of unknown, nonlinear,
    %multi-input/multi-output dynamics for the MIMO-version of AI-MOLE.
    
    properties
        number_states
        number_inputs
        number_samples_per_trial
        gp_cell
        X_cell
        u_cell
        output_matrix
        VT
    end
    
    methods
        function obj = CMIMOSSGP(number_states, ...
                                 number_inputs, ...
                                 number_samples_per_trial,...
                                 output_matrix)
            obj.number_states = number_states;
            obj.number_inputs = number_inputs;
            obj.number_samples_per_trial   = number_samples_per_trial;
            obj.gp_cell = cell(0,0);
            obj.X_cell  = cell(0,0);
            obj.u_cell  = cell(0,0);
            obj.output_matrix = output_matrix;
        end
        function [y, X] = predict(obj, u, x0)
            % Sizes
            R = obj.number_inputs;
            N = length(u)/R;
            M = obj.number_states;
            Q = size(obj.output_matrix, 1);
            % Allocate
            y = zeros(Q*N,1);
            X = zeros(M, N+1);
            % Init
            x = x0;
            X(:, 1) = x;
            % Iteratively compute predictions
            for n = 1:N
                % Fetch input sample
                un = u(1+R*(n-1):R*n,1);
                % Regression vector
                v  = obj.regression_vector(x, un);
                % Iteratively predict states
                for m = 1:M
                    [x(m,1)] = predict(obj.gp_cell{m,1}, v');
                end
                % Save values
                X(:, n+1) = x;
                y(1+Q*(n-1):Q*n,1)    = obj.output_matrix * x;
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
                opt_options        = statset('fitrgp');
                opt_options.TolFun = 1e-4;
                opt_options.TolX   = 1e-6;
                opt_options.UseParallel = true;
                % GP Training
                obj.gp_cell{m,1} = fitrgp(obj.VT', zT_cell{m,1}, 'BasisFunction', 'none', 'KernelFunction', 'ardsquaredexponential', 'OptimizerOptions', opt_options);
            end
        end
        function [P, A_cell, B_cell] = linearize_at_input_trajectory(obj, x0, u)
            % Sizes
            N = obj.number_samples_per_trial;
            M = obj.number_states;
            R = obj.number_inputs;
            % Allocate dynamic cells
            A_cell = cell(N,1);
            B_cell = cell(N,1);
            % Iteratively compute state-space linearizations
            x = x0;
            for n = 1:N
                % Current input sample
                un = u(1+R*(n-1):R*n, 1);
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
            Px = zeros(N*M, N*R);
            % Iteratively compute state Markov parameters
            for n = 1:N
                % Retrieve current state and input matrix
                A = A_cell{n,1};
                B = B_cell{n,1};
                if(n == 1)
                    Px(1:M, 1:R) = B;
                else
                    % Iteratively update the matrix columns of Px
                    for p = 1:n-1
                        Px(1+M*(n-1):M*n, 1+R*(p-1):R*p) = A*Px(1+M*(n-2):M*(n-1), 1+R*(p-1):R*p);
                    end
                    % Add the B entry
                    Px(1+M*(n-1):M*n, 1+R*(n-1):R*n) = B;
                end
            end
            % Compute P by multiplication with C
            C = obj.output_matrix;
            Q = size(C,1);
            P = zeros(Q*N, R*N);
            for n = 1:N
                P(1+Q*(n-1):Q*n,:) = C*Px(1+M*(n-1):M*n,:);
            end
        end
        function [A, B]  = linearize_at_regression_vector(obj, v)
            % Sizes
            n = obj.number_states;
            r = obj.number_inputs;
            % Allocate
            A = zeros(n, n);
            B = zeros(n, r);
            % Iteratively compute rows of A and B
            for m = 1:n
                % Retrieve current GP
                gp = obj.gp_cell{m, 1};
                % Retrieve alpha vector
                a  = gp.Impl.AlphaHat;
                % Kernel Function
                kfcn = gp.Impl.Kernel.makeKernelAsFunctionOfXNXM(gp.Impl.ThetaHat);
                % Squared inverse of length scale
                l = gp.KernelInformation.KernelParameters(1:n+r);
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
            r = obj.number_inputs;
            J = length(obj.X_cell);
            % Allocate regression training matrix
            VT = zeros(m+r, N*J);
            % Iteratively compute regression matrices
            for j = 1:J
                VT(:, 1+N*(j-1):N*j) = obj.regression_matrix(obj.X_cell{j,1}, obj.u_cell{j,1});
            end
        end
        function V = regression_matrix(obj, X, u)
            % Sizes
            N = obj.number_samples_per_trial;
            m = obj.number_states;
            r = obj.number_inputs;
            % Allocate regression matrix
            V = zeros(m+r, N);
            % Iteratively compute regression vectors
            for n = 1:N
                V(:, n) = obj.regression_vector(X(:,n), u(1+r*(n-1):r*n,1));
            end
        end
        function v = regression_vector(obj, x, u)
            v = [x; u];
        end
    end
end


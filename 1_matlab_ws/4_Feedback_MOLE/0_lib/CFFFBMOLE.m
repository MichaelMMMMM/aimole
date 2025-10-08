classdef CFFFBMOLE < handle
    %FFFB_MOLE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dyn_func
        number_inputs
        number_states
        output_matrix
        number_trials_for_training
    end
    
    methods
        function obj = CFFFBMOLE(dyn_func, ...
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
        
        function [ev, ec, yc, uc] = run_fffb_mole(obj, ...
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
            Kc = cell(N,1);
            for n = 1:N
                Kc{n} = zeros(R, M);
            end
            Xref = zeros(M, N);
            
            % Allocate return objects
            ev = zeros(J,1);
            ec = cell(J,1);
            Xc = cell(J,1);
            yc = cell(J,1);
            uc = cell(J,1);

            % Iterate
            for j = 1:J
                % Run trial
                control_func = @(x, n)obj.control_policy(x, n, u, Xref, Kc);
                [y, X] = obj.dyn_func(control_func, N);
                
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

                % -- ILC Feedforward Update
                % NO-ILC weights
                [W, S] = obj.ilc_weighting(P);
                % Learning gain matrix
                L = (P'*W*P+S)\P'*W;
                % Tracking error
                e = r - y;
                % Input update
                u = u + L*e;
                
                % -- LQR Feedback Update
                % Linearize GP-model at next-trial input trajectory
                [P, Ac, Bc] = gp_model.linearize_at_input_trajectory(zeros(M,1), u);
                % Predict next-trial state trajectory
                [~, Xref] = gp_model.predict(u, zeros(M,1));
                % LQR Weighting
                [Qfb, Rfb] = obj.lqr_weighting(P);
                % LQR Design
                Kc  = obj.ltv_lqr_design(Ac, Bc, Qfb, Rfb);

                % Save error data
                ec{j,1} = e;
                ev(j,1) = norm(e);
            end
        end
        function [ev, ec, yc, uc] = run_vanilla_mole(obj, ...
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
                control_func = @(x, n)obj.feedforward_policy(x, n, u);
                [y, X] = obj.dyn_func(control_func, N);

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

                % -- ILC Feedforward Update
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
        function un = feedforward_policy(obj, ~, time_sample, u)
            % Sizes
            R = obj.number_inputs;
            % Time sample
            n = time_sample;
            % Fetch feedforward sample
            un = u(1+R*(n-1):R*n);
        end
        function un = control_policy(obj, x, time_sample, u, Xref, Kc)
            % Sizes
            R = obj.number_inputs;
            % Time sample
            n = time_sample;
            % Fetch Feedforward input 
            u_ff = u(1+R*(n-1):R*n);
            % Fetch current feedback matrix
            K    = Kc{n};
            % Fetch current state reference
            xref = Xref(:, n);
            % Compute Feedback input
            u_fb = -K*(x-xref);
            % Compute current input
            un = u_ff + u_fb;
        end
        function Kc = ltv_lqr_design(obj, Ac, Bc, Q, R)
            % Sizes
            N = length(Ac);
            % Allocate
            Kc = cell(N,1);
            S  = Q;
            % Iteratively compute feedback controllers
            for n = 1:N
                idx = N+1-n;
                % Fetch System Matrices
                A = Ac{idx,1};
                B = Bc{idx,1};
                % Compute K
                K = (B'*Q*B+R)\B'*S*A;
                % Save K
                Kc{idx,1} = K;
                % Update S
                S = A'*(S-S*B/(B'*S*B+R)*B'*S)*A+Q;
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
            S  = diag(repmat(sv, N,1));
        end
        function [Q, R] = lqr_weighting(obj, P)
            % Sizes
            O = size(obj.output_matrix, 1);
            R = obj.number_inputs;
            C = obj.output_matrix;
            % Determine weights
            Pc = obj.P_2_subP(P);
            wv = zeros(O, 1);
            for o = 1:O
                wv(o,1) = 1/norm(cell2mat(Pc(o, :)));
            end
            W = diag(wv);
            Q = C'*W*C;

            sv = zeros(R,1);
            for r = 1:R
                sv(r,1) = norm(cell2mat(Pc(:, r)));
            end
            R = diag(sv);
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


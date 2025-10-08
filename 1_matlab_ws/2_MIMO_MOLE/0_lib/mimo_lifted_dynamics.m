function P = mimo_lifted_dynamics(A, B, C, N)
    % Sizes
    m = size(C, 1);     % Number of outputs
    r = size(B, 2);     % Number of inputs
    % Allocate
    P_cell = cell(N+1,1);
    P_cell{1,1} = zeros(m, r);  % Zero matrix
    % Initialize Markov parameter matrix
    Pn     = B;
    % Iteratively compute Markov parameter matrices
    for n = 1:N
        % Save current Markov parameter matrix
        P_cell{1+n,1} = C*Pn;
        % Update markov parameter matrix
        Pn = A*Pn;
    end
    % Create selection matrix
    selection_matrix = tril(toeplitz(1:N))+1;
    % Create lifted matrix
    P = cell2mat(P_cell(selection_matrix));
end


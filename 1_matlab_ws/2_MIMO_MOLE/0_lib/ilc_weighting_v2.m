function [W, S] = ilc_weighting_v2(P, number_samples, number_outputs, number_inputs)
    % Sizes
    N = number_samples;
    Q = number_outputs;
    R = number_inputs;
    % Determine weights
    Pc = P2subP(P, R, Q);
    wv = zeros(Q, 1);
    for q = 1:Q
        wv(q,1) = 20/norm(cell2mat(Pc(q, :)));
    end
    W  = diag(repmat(wv, N,1));
    sv = zeros(R,1);
    for r = 1:R
        sv(r,1) = norm(cell2mat(Pc(:, r)));
    end
    S  = 0.1*diag(repmat(sv, N,1));
end


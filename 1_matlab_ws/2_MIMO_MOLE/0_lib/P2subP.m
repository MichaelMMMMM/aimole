function Pc = P2subP(P, number_inputs, number_outputs)
    % Sizes
    Q = number_outputs;
    R = number_inputs;
    % Allocate
    Pc = cell(Q, R);
    for q = 1:Q
        for r = 1:R
            Pc{q,r} = P(q:Q:end, r:R:end);
        end
    end
end


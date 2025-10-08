function xd = dx_single_pendulum(~, x, u)
    % Extract states
    a  = x(1,1);
    ad = x(2,1);

    % Compute Angular accelerations
    add = (15625000*u)/79883 - (156250*ad)/79883 - (8294355*sin(a))/159766;

    % State derivative vector
    xd = [ad; add];
end


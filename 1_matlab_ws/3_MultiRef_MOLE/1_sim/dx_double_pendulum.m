function xd = dx_double_pendulum(~, x, u)
    % Extract states
    a  = x(1,1);
    ad = x(2,1);
    b  = x(3,1);
    bd = x(4,1);

    % Compute Angular accelerations
    add = -(103645000000*bd - 103645000000*ad + 10364500000000*u - 4283908838550*sin(a) + 878575438080*sin(a + 2*b) + 18001392768*ad^2*sin(2*b) + 59295000000*bd*cos(b) + 31465627008*ad^2*sin(b) + 31465627008*bd^2*sin(b) + 62931254016*ad*bd*sin(b))/(18001392768*cos(2*b) - 84398105267);
    bdd = (593220859375*bd - 207290000000*ad + 20729000000000*u + 5529809435940*sin(a + b) - 8567817677100*sin(a) - 5906884175460*sin(a - b) + 1757150876160*sin(a + 2*b) + 72005571072*ad^2*sin(2*b) + 36002785536*bd^2*sin(2*b) - 237180000000*ad*cos(b) + 355770000000*bd*cos(b) + 23718000000000*u*cos(b) + 297261061956*ad^2*sin(b) + 62931254016*bd^2*sin(b) + 72005571072*ad*bd*sin(2*b) + 125862508032*ad*bd*sin(b))/(36002785536*cos(2*b) - 168796210534);

    % State derivative vector
    xd = [ad; add; bd; bdd];
end


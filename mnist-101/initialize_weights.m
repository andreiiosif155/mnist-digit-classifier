function [matrix] = initialize_weights(L_prev, L_next)
    epsilon = sqrt(6) / sqrt(L_prev + L_next);
    matrix = rand(L_next, L_prev + 1) .* 2 * epsilon - epsilon;
    % initializam aleator matricea
end
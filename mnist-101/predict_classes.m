function [classes] = predict_classes(X, weights, ...
        input_layer_size, hidden_layer_size, ...
        output_layer_size)

    Theta1 = reshape(weights(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(weights(hidden_layer_size * (input_layer_size + 1) + 1:end), output_layer_size, (hidden_layer_size + 1));
    m = size(X, 1);
    a1 = [ones(m, 1) X];
    z2 = a1 * Theta1'; % Se calculeaza activarea stratului ascuns
    a2 = a2 = sigmoid(z2);
    a2 = [ones(m, 1) a2];
    z3 = a2 * Theta2'; % Se calculeaza activarea stratului de iesire
    a3 = sigmoid(z3);

    % Se determina clasa prezisa ca indicele elementului maxim de pe fiecare linie
    [~, classes] = max(a3, [], 2);
end
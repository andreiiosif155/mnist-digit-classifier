function [J, grad] = cost_function(params, X, y, lambda, ...
        input_layer_size, hidden_layer_size, ...
        output_layer_size)

    % Se reconstruiesc matricile de greutati Theta1 si Theta2 din vectorul params
    Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(params(hidden_layer_size * (input_layer_size + 1) + 1:end), output_layer_size, (hidden_layer_size + 1));

    m = size(X, 1);

    a1 = [ones(m, 1) X]; % Adaugam bias unitatii de input
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    a2 = [ones(m, 1), a2]; % Adaugam bias stratului ascubs
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    h = a3;

    % Se creeaza o matrice Y extinsa pentru cross-entropy
    Y = eye(output_layer_size)(y, :);

    d3 = a3 - Y; % Eroare la stratul de iesire
    d2 = (d3 * Theta2)(:, 2:end) .* (a2(:, 2:end) .* (1 - a2(:, 2:end))); % Eroare la stratul ascuns

    % Se calculeaza matricile Delta
    Delta1 = d2' * a1;
    Delta2 = d3' * a2;

    % Gradientul fara regularizare
    Theta1_grad = Delta1 / m;
    Theta2_grad = Delta2 / m;

    % Gradient cu regularizare fara coloana de bias
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

    % Functia de cost
    J = (1 / m) * sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h)));

    % Regularizare pentru cost
    reg = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));
    J = J + reg;

    % Vectorizam gradientul
    grad = [Theta1_grad(:); Theta2_grad(:)];
end
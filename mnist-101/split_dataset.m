function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
    m = size(X, 1);
    P = randperm(m); % Se genereaza o permutare aleatoare a indicilor 1..m
    X = X(P, :); % Se amesteca liniile din X
    y = y(P, :); % Se amesteca valorile corespunzatoare din y

    % Se calculeaza numarul de exemple pentru setul de antrenare
    m_train = floor(percent * m);

    % Se selecteaza exemplele pentru antrenare
    X_train = X(1:m_train, :);  
    y_train = y(1:m_train, :);

    % Se selecteaza exemplele pentru testare
    X_test = X(m_train + 1:end, :);
    y_test = y(m_train + 1:end, :);

end
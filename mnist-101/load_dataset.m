function [X, y] = load_dataset(path)
    data = load(path); % Incarca fisierul .mat specificat prin path
    X = data.X;
    y = data.y;
end
% Parámetros de la red
inputSize = 4;
hiddenSize = 20;
outputSize = 3;
learningRate = 0.3;
epochs = 20;

% Función de activación ReLU
relu = @(x) max(0, x);
relu_derivative = @(x) (x > 0);

% Cargar los datos de entrenamiento desde un archivo CSV
data = csvread('/MATLAB Drive/Archivos/irisbin.csv');

% Dividir los datos en entradas y salidas
inputs = data(:, 1:inputSize)
targets = data(:, inputSize + 1:inputSize + outputSize)
% Calcula la media y la desviación estándar de los datos de entrada.
mu = mean(inputs, 1);
sigma = std(inputs, 1);

% Normaliza los datos de entrada con Z-score.
normalized_inputs = (inputs - mu) ./ sigma;

% Inicialización de los pesos
W1 = randn(hiddenSize, inputSize);
W2 = randn(outputSize, hiddenSize);

% Parámetros de regularización (puedes ajustar estos valores)
lambda1 = 0.2; % Para W1
lambda2 = 0.1; % Para W2

% Entrenamiento
for epoch = 1:epochs
    total_error = 0;

    for i = 1:size(normalized_inputs, 1)
        % Forward
        hidden_activation = normalized_inputs(i, :) * W1';
        hidden_output = relu(hidden_activation);
        final_activation = hidden_output * W2';
        final_output = relu(final_activation);

        % Error
        error = sum((targets(i, :) - final_output) .^ 2) / 2;
        total_error = total_error + error;

        % Retropropagación
        final_error = (final_output - targets(i, :)) .* relu_derivative(final_activation);
        hidden_error = (final_error * W2) .* relu_derivative(hidden_activation);

        % Actualización de pesos
        delta_W2 = learningRate * final_error' * hidden_output;
        delta_W1 = learningRate * hidden_error' * normalized_inputs(i, :);

        W2 = W2 - delta_W2;
        W1 = W1 - delta_W1;
    end

    fprintf('Época %d - Error promedio: %f\n', epoch, total_error / size(normalized_inputs, 1));
end



% Generar predicciones
new_inputs = [7.7 2.6 6.9 2.3];
normalized_new_inputs = (new_inputs - mu) ./ sigma;
hidden_activation = normalized_new_inputs * W1';
hidden_output = relu(hidden_activation);
final_activation = hidden_output * W2';
predictions = relu(final_activation);
% Aplicar Softmax a las predicciones
predictions_softmax = exp(predictions - max(predictions, [], 2)) ./ sum(exp(predictions - max(predictions, [], 2)), 2);
disp('Predicciones (Softmax):');
disp(predictions);
% Concatenar las matrices horizontalmente
combined_data = [inputs, targets];
% y una capa de salida de 3 neuronas.
Net = feedforwardnet(20,'trainlm');
% Suponiendo que ya has creado y configurado tu red neuronal 'Net'
[Net, tr] = train(Net, combined_data(:, 1:4)', combined_data(:, 5:7)');

                                    % Supongamos que 'input_data' es un vector de 4 valores
input_data = [5, 3.3, 1.4, 0.2];

                                    % Asegúrate de que 'input_data' esté en la forma correcta (como una columna)
input_data = input_data(:);

% Pasa 'input_data' a través de la red neuronal

output_data = Net(input_data);
output_data_rounded = round(output_data);
output_data_rounded = output_data_rounded';
% Reemplazar cualquier valor diferente de -1 o 1 con esos valores
output_data_rounded(output_data_rounded ~= -1 & output_data_rounded ~= 1) = 1;
if isequal(output_data_rounded, [-1, -1, 1])
    disp('La especie predicha es Setosa');
elseif isequal(output_data_rounded, [-1, 1, -1])
    disp('La especie predicha es Versicolor');
elseif isequal(output_data_rounded, [1, -1, -1])
    disp('La especie predicha es Virginica');
else
    disp('No se puede determinar la especie');
end

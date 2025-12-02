clear; clc; close all;

%% PARAMETERS
DATA_TRAIN      = "reduced_train.csv";
DATA_VAL        = "reduced_val.csv";

NUM_EPOCHS      = 150;
NUM_CLUSTERS    = 4;          % cluster number = More/less rules
EXPONENT        = 2.2;        % fuzzification coefficient (higher coefficient more fuzzy)
MIN_IMPROVEMENT = 1e-5;       % Minimum Improvement (Lower number for lower improvement dont go past 8)
MAX_ITER        = 150;        %Amount of itterations allowed (Keep same as epoches or lower)

%% LOAD DATA
trainData = readmatrix(DATA_TRAIN);
valData   = readmatrix(DATA_VAL);

Xin  = trainData(:,1:end-1);
Xout = trainData(:,end);

if any(isnan(trainData(:))) || any(isnan(valData(:)))
    error("Missing values detected.");
end

%% FIS INITIALISATION — FCM CLUSTERING

%% FIS INITIALISATION
disp("FCM")
optsFIS = genfisOptions("FCMClustering");
optsFIS.NumClusters       = NUM_CLUSTERS;
optsFIS.Exponent          = EXPONENT;
optsFIS.MinImprovement    = MIN_IMPROVEMENT;
optsFIS.MaxNumIteration   = MAX_ITER;

initFIS = genfis(Xin, Xout, optsFIS);

%% TRAINING
optsTrain = anfisOptions("InitialFIS", initFIS);
optsTrain.EpochNumber = NUM_EPOCHS;
optsTrain.ValidationData = valData;
optsTrain.DisplayANFISInformation = 0;
optsTrain.DisplayErrorValues = 1;

[trainedFIS, trainError, ~, valError] = anfis(trainData, optsTrain);

%% EVALUATION
Y_pred = evalfis(trainedFIS, valData(:,1:end-1));
Y_true = valData(:,end);

rmse = sqrt(mean((Y_pred - Y_true).^2));
disp("Validation RMSE: " + rmse);

%% PLOT
trainError = clean_error(trainError);
valError   = clean_error(valError);

epochs_train = 1:length(trainError);
epochs_val   = 1:length(valError);

% TRAINING CURVE
figure;

plot(epochs_train, trainError, 'LineWidth', 1.5); hold on;

if all(~isnan(valError))
    plot(epochs_val, valError, 'LineWidth', 1.5);
end

xlabel('Epoch');
ylabel('Root Mean Square Error');
title('ANFIS Training and Validation Error');
legend('Training Error','Validation Error');
grid on;

% FITTED OUTPUT
figure;
plot(Y_true, 'LineWidth', 1.3); hold on;
plot(Y_pred, 'LineWidth', 1.3);
xlabel('Sample Index');
ylabel('Output Value');
title('True vs Predicted Output (Fitted Curve)');
legend('True','Predicted');
grid on;

% ERROR METRICS
abs_err = abs(Y_pred - Y_true);
sq_err  = (Y_pred - Y_true).^2;

figure;

subplot(3,1,1);
plot(abs_err, 'LineWidth', 1.2);
ylabel('Absolute Error');
title('Prediction Error Metrics');
grid on;

subplot(3,1,2);
plot(sq_err, 'LineWidth', 1.2);
ylabel('Squared Error');
grid on;

subplot(3,1,3);
histogram(Y_pred - Y_true, 25);
xlabel('Residual');
ylabel('Frequency');
grid on;

function e = clean_error(raw)
    if isempty(raw)
        e = nan; 
        return;
    end
    if iscell(raw)
        try
            raw = cell2mat(raw);
        catch
            raw = [raw{:}];
        end
    end
    if isnumeric(raw) && isscalar(raw)
        e = double(raw);
        return;
    end
    if isnumeric(raw)
        e = double(raw(:));
        return;
    end
    if isa(raw, "sugfis") || isa(raw, "mamfis")
        e = nan; 
        return;
    end
    error("Unrecognised error format returned by ANFIS.");
end

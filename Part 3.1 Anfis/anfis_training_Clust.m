clear; clc; close all;

%% PARAMETERS
DATA_TRAIN      = "reduced_train.csv";
DATA_VAL        = "reduced_val.csv";

NUM_EPOCHS      = 150;        % reduce for speed
CLUSTER_RANGE   = 0.35;       % larger = fewer clusters (faster)
SQUASH_FACTOR   = 0.85;          %Controls how anfis sqaushes data far from point (Higher more sqaush)
ACCEPT_RATIO    = 0.45; %Accept clusters threshhold (higher val = less clusters)
REJECT_RATIO    = 0.05;  %Boundary for rejecting clusters (Higher val more rigorous)

%% LOAD DATA
trainData = readmatrix(DATA_TRAIN);
valData   = readmatrix(DATA_VAL);

Xin  = trainData(:,1:end-1);
Xout = trainData(:,end);

if any(isnan(trainData(:))) || any(isnan(valData(:)))
    error("Missing values detected.");
end

%% FIS INITIALISATION
disp("Subtractive")
optsFIS = genfisOptions("SubtractiveClustering");
optsFIS.ClusterInfluenceRange = CLUSTER_RANGE;
optsFIS.SquashFactor          = SQUASH_FACTOR;
optsFIS.AcceptRatio           = ACCEPT_RATIO;
optsFIS.RejectRatio           = REJECT_RATIO;

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
fplot(@(x) interp1(1:length(abs_err), abs_err, x, "linear"), ...
      [1 length(abs_err)], 'LineWidth', 1.2);
ylabel('Absolute Error');
title('Prediction Error Metrics');
grid on;

subplot(3,1,2);
fplot(@(x) interp1(1:length(sq_err), sq_err, x, "linear"), ...
      [1 length(sq_err)], 'LineWidth', 1.2);
ylabel('Squared Error');
grid on;

subplot(3,1,3);
histogram(Y_pred - Y_true, 25);
xlabel('Residual');
ylabel('Frequency');
grid on;

function e = clean_error(raw)
    % Converts ANFIS error output into a numeric vector

    if isempty(raw)
        e = nan; 
        return;
    end

    % unwrap cell → numeric
    if iscell(raw)
        try
            raw = cell2mat(raw);
        catch
            raw = [raw{:}];
        end
    end

    % if scalar numeric, expand
    if isnumeric(raw) && isscalar(raw)
        e = double(raw);
        return;
    end

    % if numeric vector
    if isnumeric(raw)
        e = double(raw(:));
        return;
    end

    % if ANFIS mistakenly returned a FIS object
    if isa(raw, "sugfis") || isa(raw, "mamfis")
        e = nan; 
        return;
    end

    error("Unrecognised error format returned by ANFIS.");
end

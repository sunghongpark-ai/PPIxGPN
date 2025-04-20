%% Data Setting

% Protein expression
Xdata % p by n matrix

% Real diagnosis
Ydata % n by 4 matrix

% Data split
idx_train; dataset.Xtrain = Xdata(:,idx_train); dataset.Ytrain = Ydata(:,idx_train);
idx_valid; dataset.Xvalid = Xdata(:,idx_valid); dataset.Yvalid = Ydata(:,idx_valid);
idx_test ; dataset.Xtest = Xdata(:,idx_test)  ; dataset.Ytest = Ydata(:,idx_test)  ;

% PPI network
dataset.Lppi = PPInetwork(ppi_data); num_protein = size(ppi_data,1);

%% Parameter Setting

% Propagation parameter
parameter.Uppi = ParamInit(num_protein,1);

% Estimation parameter
parameter.Babt = ParamInit(num_protein,0);
parameter.Bgfa = ParamInit(num_protein,0);
parameter.Bnfl = ParamInit(num_protein,0);
parameter.Btau = ParamInit(num_protein,0);

% Model parameter
parameter.epoch = 1000;
parameter.rate = 0.001;
parameter.gamma = 0.01;

%% Model Implementation

% Model training
model_param = ModelTrain(dataset,parameter);

% Risk prediction
pred_risk = RiskPredict(dataset,parameter,model_param);

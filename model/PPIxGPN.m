%% Data Setting

num_train = length(idx_train); Xtrain = Xdata(idx_train,:); Ytrain = Ydata(idx_train,:);
num_valid = length(idx_valid); Xvalid = Xdata(idx_valid,:); Yvalid = Ydata(idx_valid,:);
num_test = length(idx_test)  ; Xtest = Xdata(idx_test,:)  ; Ytest = Ydata(idx_test,:)  ;

% PPI network
Wppi = Wppi.*(Wppi>=0.4); Wppi(Wppi>0) = 1./(1+exp((-1)*(Wppi(Wppi>0)-mean(Wppi(Wppi>0)))./std(Wppi(Wppi>0))));
diagW = 1./sqrt(sum(Wppi,2)); diagW(isinf(diagW)==1) = 0; Lppi = eye(num_protein)-(diag(diagW)*Wppi*diag(diagW));

%% Parameter Setting

% Propagation parameter
Uppi = mu*ones(num_protein,1); szUppi = size(Uppi);

% Estimation parameter
Babt = zeros(num_protein,1)  ; szBabt = size(Babt); Babt = (2*rand(szBabt)-1)*sqrt(6/sum(szBabt));
Bgfa = zeros(num_protein,1)  ; szBgfa = size(Bgfa); Bgfa = (2*rand(szBgfa)-1)*sqrt(6/sum(szBgfa));
Bnfl = zeros(num_protein,1)  ; szBnfl = size(Bnfl); Bnfl = (2*rand(szBnfl)-1)*sqrt(6/sum(szBnfl));
Btau = zeros(num_protein,1)  ; szBtau = size(Btau); Btau = (2*rand(szBtau)-1)*sqrt(6/sum(szBtau));

num_epoch = 1000; learn_rate = 0.001; gamma = 0.001;

num_param = prod(szUppi)+prod(szBabt)+prod(szBgfa)+prod(szBnfl)+prod(szBtau);
set_param = AdamInit(num_param,learn_rate);

W = [Uppi(:);Babt(:);Bgfa(:);Bnfl(:);Btau(:)];

EtrainABT = zeros(num_epoch,1); EvalidABT = zeros(num_epoch,1);
EtrainGFA = zeros(num_epoch,1); EvalidGFA = zeros(num_epoch,1);
EtrainNFL = zeros(num_epoch,1); EvalidNFL = zeros(num_epoch,1);
EtrainTAU = zeros(num_epoch,1); EvalidTAU = zeros(num_epoch,1);

weight = cell(num_epoch,1);

%% Model Training

for idx_epoch = 1:num_epoch

    weight{idx_epoch,1} = W;

    start = 1                 ; finish = prod(szUppi)       ; Uppi = reshape(W(start:finish),szUppi);
    start = start+prod(szUppi); finish = finish+prod(szBabt); Babt = reshape(W(start:finish),szBabt);
    start = start+prod(szBabt); finish = finish+prod(szBgfa); Bgfa = reshape(W(start:finish),szBgfa);
    start = start+prod(szBgfa); finish = finish+prod(szBnfl); Bnfl = reshape(W(start:finish),szBnfl);
    start = start+prod(szBnfl); finish = finish+prod(szBtau); Btau = reshape(W(start:finish),szBtau);

    % FeedForward
    Dppi = diag(Uppi);
    Ztrain = (Dppi+Lppi)\Dppi*Xtrain           ; Zvalid = (Dppi+Lppi)\Dppi*Xvalid           ; Ztest = (Dppi+Lppi)\Dppi*Xtest           ;
    PtrainABT = 1./(1+exp((-1)*(Babt'*Ztrain))); PvalidABT = 1./(1+exp((-1)*(Babt'*Zvalid))); PtestABT = 1./(1+exp((-1)*(Babt'*Ztest)));
    PtrainGFA = 1./(1+exp((-1)*(Bgfa'*Ztrain))); PvalidGFA = 1./(1+exp((-1)*(Bgfa'*Zvalid))); PtestGFA = 1./(1+exp((-1)*(Bgfa'*Ztest)));
    PtrainNFL = 1./(1+exp((-1)*(Bnfl'*Ztrain))); PvalidNFL = 1./(1+exp((-1)*(Bnfl'*Zvalid))); PtestNFL = 1./(1+exp((-1)*(Bnfl'*Ztest)));
    PtrainTAU = 1./(1+exp((-1)*(Btau'*Ztrain))); PvalidTAU = 1./(1+exp((-1)*(Btau'*Zvalid))); PtestTAU = 1./(1+exp((-1)*(Btau'*Ztest)));

    % Loss
    EtrainABT(idx_epoch) = (-1/num_train)*((YtrainABT'*log(PtrainABT'))+((1-YtrainABT')*log((1-PtrainABT')))); EvalidABT(idx_epoch) = (-1/num_valid)*((YvalidABT'*log(PvalidABT'))+((1-YvalidABT')*log((1-PvalidABT'))));
    EtrainGFA(idx_epoch) = (-1/num_train)*((YtrainGFA'*log(PtrainGFA'))+((1-YtrainGFA')*log((1-PtrainGFA')))); EvalidGFA(idx_epoch) = (-1/num_valid)*((YvalidGFA'*log(PvalidGFA'))+((1-YvalidGFA')*log((1-PvalidGFA'))));
    EtrainNFL(idx_epoch) = (-1/num_train)*((YtrainNFL'*log(PtrainNFL'))+((1-YtrainNFL')*log((1-PtrainNFL')))); EvalidNFL(idx_epoch) = (-1/num_valid)*((YvalidNFL'*log(PvalidNFL'))+((1-YvalidNFL')*log((1-PvalidNFL'))));
    EtrainTAU(idx_epoch) = (-1/num_train)*((YtrainTAU'*log(PtrainTAU'))+((1-YtrainTAU')*log((1-PtrainTAU')))); EvalidTAU(idx_epoch) = (-1/num_valid)*((YvalidTAU'*log(PvalidTAU'))+((1-YvalidTAU')*log((1-PvalidTAU'))));
    
    % BackProp
    dEdBabt = (1/num_train)*Ztrain*(PtrainABT-YtrainABT')'; gradBabt = dEdBabt + 2*gamma*Babt;
    dEdBgfa = (1/num_train)*Ztrain*(PtrainGFA-YtrainGFA')'; gradBgfa = dEdBgfa + 2*gamma*Bgfa;
    dEdBnfl = (1/num_train)*Ztrain*(PtrainNFL-YtrainNFL')'; gradBnfl = dEdBnfl + 2*gamma*Bnfl;
    dEdBtau = (1/num_train)*Ztrain*(PtrainTAU-YtrainTAU')'; gradBtau = dEdBtau + 2*gamma*Btau;
    dEdZ = (1/num_train)*((Babt*(PtrainABT-YtrainABT'))+(Bgfa*(PtrainGFA-YtrainGFA'))+(Bnfl*(PtrainNFL-YtrainNFL'))+(Btau*(PtrainTAU-YtrainTAU')));
    dEdUppi = dEdZ*Xtrain'/(Dppi+Lppi)*(eye(num_protein)-((Dppi+Lppi)\Dppi)).*eye(num_protein)*ones(1,num_protein)';
    gradUppi = dEdUppi + 2*gamma*Uppi;

    G = [gradUppi(:);gradBabt(:);gradBgfa(:);gradBnfl(:);gradBtau(:)];

    [W,set_param] = WeightUpdate(W,G,set_param);

end % End epoch
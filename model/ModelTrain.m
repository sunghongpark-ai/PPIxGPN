function weight = ModelTrain(dataset,parameter)

Xtrain = dataset.Xtrain; Ytrain = dataset.Ytrain; Ntrain = size(Ytrain,1);
Xvalid = dataset.Xvalid; Yvalid = dataset.Yvalid; Nvalid = size(Yvalid,1);

YtrainABT = Ytrain(:,1); YtrainGFA = Ytrain(:,2); YtrainNFL = Ytrain(:,3); YtrainTAU = Ytrain(:,4);
YvalidABT = Yvalid(:,1); YvalidGFA = Yvalid(:,2); YvalidNFL = Yvalid(:,3); YvalidTAU = Yvalid(:,4);

Lppi = dataset.Lppi; Nprotein = size(Lppi,1);

Uppi = parameter.Uppi; Babt = parameter.Babt; Bgfa = parameter.Bgfa; Bnfl = parameter.Bnfl; Btau = parameter.Btau;
Wparam = [Uppi(:);Babt(:);Bgfa(:);Bnfl(:);Btau(:)];
Wsize = [size(Uppi);size(Babt);size(Bgfa);size(Bnfl);size(Btau)];

num_epoch = parameter.epoch; learn_rate = parameter.rate; reg_coeff = parameter.gamma; adam_param = AdamInit(Wparam,learn_rate);

Wepoch = cell(num_epoch,1);

EtrainABT = zeros(num_epoch,1); EvalidABT = zeros(num_epoch,1);
EtrainGFA = zeros(num_epoch,1); EvalidGFA = zeros(num_epoch,1);
EtrainNFL = zeros(num_epoch,1); EvalidNFL = zeros(num_epoch,1);
EtrainTAU = zeros(num_epoch,1); EvalidTAU = zeros(num_epoch,1);

for idx_epoch = 1:num_epoch

    Wepoch{idx_epoch,1} = Wparam; [Uppi,Babt,Bgfa,Bnfl,Btau] = ResizeParam(Wparam,Wsize);
    
    % ForwardPropagation
    Dppi = diag(Uppi);
    Ztrain = (Dppi+Lppi)\Dppi*Xtrain           ; Zvalid = (Dppi+Lppi)\Dppi*Xvalid           ;
    PtrainABT = 1./(1+exp((-1)*(Babt'*Ztrain))); PvalidABT = 1./(1+exp((-1)*(Babt'*Zvalid)));
    PtrainGFA = 1./(1+exp((-1)*(Bgfa'*Ztrain))); PvalidGFA = 1./(1+exp((-1)*(Bgfa'*Zvalid)));
    PtrainNFL = 1./(1+exp((-1)*(Bnfl'*Ztrain))); PvalidNFL = 1./(1+exp((-1)*(Bnfl'*Zvalid)));
    PtrainTAU = 1./(1+exp((-1)*(Btau'*Ztrain))); PvalidTAU = 1./(1+exp((-1)*(Btau'*Zvalid)));

    % LossTrainData
    EtrainABT(idx_epoch) = (-1/Ntrain)*((YtrainABT'*log(PtrainABT'))+((1-YtrainABT')*log((1-PtrainABT'))));
    EtrainGFA(idx_epoch) = (-1/Ntrain)*((YtrainGFA'*log(PtrainGFA'))+((1-YtrainGFA')*log((1-PtrainGFA'))));
    EtrainNFL(idx_epoch) = (-1/Ntrain)*((YtrainNFL'*log(PtrainNFL'))+((1-YtrainNFL')*log((1-PtrainNFL'))));
    EtrainTAU(idx_epoch) = (-1/Ntrain)*((YtrainTAU'*log(PtrainTAU'))+((1-YtrainTAU')*log((1-PtrainTAU'))));

    % LossValidData
    EvalidABT(idx_epoch) = (-1/Nvalid)*((YvalidABT'*log(PvalidABT'))+((1-YvalidABT')*log((1-PvalidABT'))));
    EvalidGFA(idx_epoch) = (-1/Nvalid)*((YvalidGFA'*log(PvalidGFA'))+((1-YvalidGFA')*log((1-PvalidGFA'))));
    EvalidNFL(idx_epoch) = (-1/Nvalid)*((YvalidNFL'*log(PvalidNFL'))+((1-YvalidNFL')*log((1-PvalidNFL'))));
    EvalidTAU(idx_epoch) = (-1/Nvalid)*((YvalidTAU'*log(PvalidTAU'))+((1-YvalidTAU')*log((1-PvalidTAU'))));
    
    % BackwardPropagation
    dEdBabt = (1/Ntrain)*Ztrain*(PtrainABT-YtrainABT')'; gradBabt = dEdBabt + 2*reg_coeff*Babt;
    dEdBgfa = (1/Ntrain)*Ztrain*(PtrainGFA-YtrainGFA')'; gradBgfa = dEdBgfa + 2*reg_coeff*Bgfa;
    dEdBnfl = (1/Ntrain)*Ztrain*(PtrainNFL-YtrainNFL')'; gradBnfl = dEdBnfl + 2*reg_coeff*Bnfl;
    dEdBtau = (1/Ntrain)*Ztrain*(PtrainTAU-YtrainTAU')'; gradBtau = dEdBtau + 2*reg_coeff*Btau;
    dEdZ = (1/Ntrain)*((Babt*(PtrainABT-YtrainABT'))+(Bgfa*(PtrainGFA-YtrainGFA'))+(Bnfl*(PtrainNFL-YtrainNFL'))+(Btau*(PtrainTAU-YtrainTAU')));
    dEdUppi = dEdZ*Xtrain'/(Dppi+Lppi)*(eye(Nprotein)-((Dppi+Lppi)\Dppi)).*eye(Nprotein)*ones(1,Nprotein)';
    gradUppi = dEdUppi + 2*reg_coeff*Uppi;

    Gparam = [gradUppi(:);gradBabt(:);gradBgfa(:);gradBnfl(:);gradBtau(:)];

    [Wparam,adam_param] = WeightUpdate(Wparam,Gparam,adam_param);

end % End epoch

Evalid = mean([EvalidABT,EvalidGFA,EvalidNFL,EvalidTAU],2)';
weight = Wepoch{find(Evalid==min(Evalid),1),1};
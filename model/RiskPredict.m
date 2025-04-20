function P = RiskPredict(dataset,parameter,model_param)

Wparam = model_param;
Wsize = [size(parameter.Uppi);size(parameter.Babt);size(parameter.Bgfa);size(parameter.Bnfl);size(parameter.Btau)];
[Uppi,Babt,Bgfa,Bnfl,Btau] = ResizeParam(Wparam,Wsize);

Z = (diag(Uppi)+dataset.Lppi)\diag(Uppi)*dataset.Xtest;
P = (1./(1+exp((-1)*([Babt,Bgfa,Bnfl,Btau]'*Z))))';
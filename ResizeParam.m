function [Uppi,Babt,Bgfa,Bnfl,Btau] = ResizeParam(param_data,param_size)

szUppi = param_size(1,:); start = 1                 ; finish = prod(szUppi)       ; Uppi = reshape(param_data(start:finish),szUppi);
szBabt = param_size(2,:); start = start+prod(szUppi); finish = finish+prod(szBabt); Babt = reshape(param_data(start:finish),szBabt);
szBgfa = param_size(3,:); start = start+prod(szBabt); finish = finish+prod(szBgfa); Bgfa = reshape(param_data(start:finish),szBgfa);
szBnfl = param_size(4,:); start = start+prod(szBgfa); finish = finish+prod(szBnfl); Bnfl = reshape(param_data(start:finish),szBnfl);
szBtau = param_size(5,:); start = start+prod(szBnfl); finish = finish+prod(szBtau); Btau = reshape(param_data(start:finish),szBtau);

function [X,cvx_status] = stabilizeROM(Ma,Me,k,p,tau)

  %run /home/users/amcclell/matlab/cvx/cvx_startup.m

  Q = eye(k);
  mu = 10^(-8); 
  cvx_begin
  cvx_solver sdpt3 
  variable P11(k,k) symmetric 
  variable P22(p,p) symmetric 
  variable P12(k,p) 
  
  minimize( norm([P12', P22],'fro') + tau*norm(P11,'fro')); 
  
  subject to 
    [P11, P12;P12', P22]-mu*eye(k+p) == semidefinite(k+p); 
    Me'*[P11, P12;P12', P22]*Ma +Ma'*[P11, P12;P12', P22]*Me ==-Q;
  cvx_end
  X = [P11;P12'];

end

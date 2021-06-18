function [X,cvx_status] = stabilizeROM(Ma,Me,k,p,tau,mu,solver,precision,opts)

  if (nargin < 6)
    mu = 1e-8;
    solver = 'sdpt3';
    opts = struct();
  elseif (nargin < 7)
    solver = 'sdpt3';
    opts = struct();
  elseif (nargin < 8)
    opts = struct();
  end

  f = fieldnames(opts);
  
  Q = eye(k);
  cvx_begin
  if strcmp(solver, 'sdpt3')
    cvx_solver sdpt3
  elseif strcmp(solver, 'sedumi')
    cvx_solver sedumi
  else
    error('*** Error: solver %s is not supported (currently only supports free solvers)', solver);
  end
  cvx_precision(precision);
  if ~isempty(f)
    c = struct2cell(opts);
    for i = 1:length(f)
      cvx_solver_settings(f{i}, c{i})
    end
  end
  variable P11(k,k) symmetric 
  variable P22(p,p) symmetric 
  variable P12(k,p)
  disp(P11)
  disp(P12)
  disp(P22)
  
  minimize( norm([P12', P22],'fro') + tau*norm(P11,'fro')); 
  
  subject to 
    [P11, P12;P12', P22]-mu*eye(k+p) == semidefinite(k+p); 
    Me'*[P11, P12;P12', P22]*Ma +Ma'*[P11, P12;P12', P22]*Me ==-Q;
cvx_end
X = [P11;P12'] ;

end

% function [Zres] = stabilizeROM(Ma,Me,k,p,mu)
%    
% addpath cvx/
% 
% Q = eye(k);
% cvx_setup
% cvxp = cvx_precision( 'best' );
% cvx_begin
% cvx_solver sdpt3
% variable Z(k,k) symmetric
% variable DZ22(p,p) symmetric
% variable DZ21(k,p)
% %
% 
% 
% minimize(  norm([DZ21;DZ22],'fro') );
% 
% 
% subject to
%   [Z,zeros(k,p);zeros(p,k+p)]+[zeros(k,k) DZ21;DZ21' DZ22]-mu*eye(k+p) == semidefinite(k+p);
%   Me'*([Z,zeros(k,p);zeros(p,k+p)]+[zeros(k,k) DZ21;DZ21' DZ22])*Ma +Ma'*([Z,zeros(k,p);zeros(p,k+p)]+[zeros(k,k) DZ21;DZ21' DZ22])*Me ==-Q;
% 
% cvx_end
% cvx_precision(cvxp)
% Zres =  [Z,zeros(k,p);zeros(p,k+p)]+[zeros(k,k) DZ21;DZ21' DZ22];
% end

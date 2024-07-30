% FTVQ: FTV with pre-computed blur parameters
%
%  x = ftvq(y,pc,mu)
%
%  y: Three channel blurred image
%  pc: Output of fpc, which has been called with acting
%      blur kernels.
%  mu: Regularization weight
%
%  x: Deconvolved Image
function x = ftvq(y,pc,mu)

out_it = 8; beta_s = 4;
beta = (100*mu) / (beta_s).^(out_it); 

%%%%%
pad = pc{1};
y = padimg(y,pad);

%%% Precompute stuff
[num2 den1 den2] = precomp(y,pc);

% Init
x = y; fct = 0.5;
for i = 1:3
  x(:,:,i) = real(ifft2(fct*num2(:,:,i) ./ (den1 + fct*den2(:,:,i))));
end;

% Find directions !
w0x = diff(x,1,2); w0y = diff(x,1,1);
w0n = repmat(sqrt(sum(w0x.^2,3)),[1 1 3]); w0x = w0x ./ w0n;
w0x(w0n == 0) = 1/sqrt(3);
w0n = repmat(sqrt(sum(w0y.^2,3)),[1 1 3]); w0y = w0y ./ w0n;
w0y(w0n == 0) = 1/sqrt(3);

for ot = 1:out_it
  % W sub-problem
  wx = diff(x,1,2)/sqrt(2); wy = diff(x,1,1)/sqrt(2);
  
  %-- Project
  wxj = sum(wx .* w0x,3); wyj = sum(wy .* w0y,3);

  %-- Shrink
  mx = max(10^-4,max(abs(wxj(:)))); mx = max(mx,max(abs(wyj(:))));
  [thr a b] = gets(beta,mx);
  
  wxj(abs(wxj) < thr) = 0; wyj(abs(wyj) < thr) = 0;
  wxj = max(0,a*abs(wxj)+b) .* sign(wxj);
  wyj = max(0,a*abs(wyj)+b) .* sign(wyj);
  
  %-- Final gradient vectors
  wx = repmat(wxj,[1 1 3]) .* w0x;
  wy = repmat(wyj,[1 1 3]) .* w0y;
    
  % X sub-problem
  for i = 1:3
    num1 = [-wx(:,1,i) -diff(wx(:,:,i),1,2) wx(:,end,i)];
    num1 = num1 + [-wy(1,:,i); -diff(wy(:,:,i),1,1); wy(end,:,i)];
    tmp = fft2(num1/sqrt(2)) + mu/beta*num2(:,:,i);
    tmp = tmp ./ (den1 + mu/beta*den2(:,:,i));
    x(:,:,i) = real(ifft2(tmp)); 
  end;
    
  
  beta = beta*beta_s;  % Beta continuation
end;


% Done
x = x(1+pad:end-pad,1+pad:end-pad,:);

function [num2 den1 den2] = precomp(y,pc)

den1 = pc{5}; den2 = pc{6};

k1f = pc{2};
k2f = pc{3};
k3f = pc{4};

num2 = conj(k1f).*fft2(y(:,:,1));
num2(:,:,2) = conj(k2f).*fft2(y(:,:,2));
num2(:,:,3) = conj(k3f).*fft2(y(:,:,3));

function y2 = padimg(y,pad)
y2 = zeros(size(y) + [2*pad 2*pad 0]);
wt = linspace(0,1,2*pad);
wtx = repmat(wt(:)',[size(y,1) 1]); wty = repmat(wt(:),[1 size(y2,2)]);
for i = 1:3
  yi = y(:,:,i);
    
  px = (1-wtx).*repmat(yi(:,end),[1 2*pad])+(wtx).*repmat(yi(:,1),[1 2*pad]);
  yi = [px(:,pad+1:end) yi px(:,1:pad)];
  
  py = (1-wty).*repmat(yi(end,:),[2*pad 1])+(wty).*repmat(yi(1,:),[2*pad 1]);
  yi = [py(pad+1:end,:); yi; py(1:pad,:)];
  
  y2(:,:,i) = yi;
end;


% Compute linear approximation to gradient shrinkage
function [thr a b] = gets(beta,mx)

v = linspace(0,mx,5000);
v2 = shrinkv23(v,beta);
idx = min(find(v2 > 0));

if length(idx) == 0
  a = 0; b = 0; thr = 10;
  return
end;

thr = v(idx);

a = (v2(end)-v2(idx)) / (v(end)-v(idx));
b = v2(end) - a*v(end);

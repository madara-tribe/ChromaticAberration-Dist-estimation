%%% Taken from the code provided by the authors of
%%% D. Krishnan, R. Fergus: "Fast Image Deconvolution using
%%% Hyper-Laplacian Priors", Proceedings of NIPS 2009. 
function w = shrinkv23(v, beta)
% solve a quartic equation
% for alpha = 2/3

epsilon = 1e-6; %% tolerance on imag part of real root  
   
k = 8/(27*beta^3);
m = ones(size(v))*k;
  
% Now use formula from
% http://en.wikipedia.org/wiki/Quartic_equation (Ferrari's method)
% running our coefficients through Mathmetica (quartic_solution.nb)
% optimized to use as few operations as possible...
        
%%% precompute certain terms
v2 = v .* v;
v3 = v2 .* v;
v4 = v3 .* v;
m2 = m .* m;
m3 = m2 .* m;
  
%% Compute alpha & beta
alpha = -1.125*v2;
beta2 = 0.25*v3;
  
%%% Compute p,q,r and u directly.
q = -0.125*(m.*v2);
r1 = -q/2 + sqrt(-m3/27 + (m2.*v4)/256);

u = exp(log(r1)/3); 
y = 2*(-5/18*alpha + u + (m./(3*u))); 
    
W = sqrt(alpha./3 + y);
  
%%% now form all 4 roots
root = zeros(size(v,1),size(v,2),4);
root(:,:,1) = 0.75.*v  +  0.5.*(W + sqrt(-(alpha + y + beta2./W )));
root(:,:,2) = 0.75.*v  +  0.5.*(W - sqrt(-(alpha + y + beta2./W )));
root(:,:,3) = 0.75.*v  +  0.5.*(-W + sqrt(-(alpha + y - beta2./W )));
root(:,:,4) = 0.75.*v  +  0.5.*(-W - sqrt(-(alpha + y - beta2./W )));
  
    
%%%%%% Now pick the correct root, including zero option.
  
%%% Clever fast approach that avoids lookups
v2 = repmat(v,[1 1 4]); 
sv2 = sign(v2);
rsv2 = real(root).*sv2;
    
%%% condensed fast version
%%%             take out imaginary                roots above v/2            but below v
root_flag3 = sort(((abs(imag(root))<epsilon) & ((rsv2)>(abs(v2)/2)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
%%% take best
w=root_flag3(:,:,1);

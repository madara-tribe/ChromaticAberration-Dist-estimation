% FTV: Fast Deconvolution with color constaints on gradients
%
%  x = ftv(y,k1,k2,k3,mu)
%
%  y: Three channel blurred image
%  k1,k2,k3: Blur kernel acting on Channels 1,2,3
%  mu: Regularization weight
%
%  x: Deconvolved Image
%
% Note: If you know k1,k2,k3 and the size of y in advance,
% consider using fpc(.) and ftvq(.). The fpc function will
% do a lot of the housekeeping a-priori, speeding up the
% actual deconvolution for each observed image.
function x = ftv(y,k1,k2,k3,mu)

pc = fpc(k1,k2,k3,size(y));
x = ftvq(y,pc,mu);
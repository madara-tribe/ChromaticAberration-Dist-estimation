% FPC: Pre-compute parameters for use with FTVQ
%
%  pc = fpc(k1,k2,k3,sz)
%
%  k1,k2,k3: Blur kernel acting on Channels 1,2,3
%  sz: Size of observed images
%
%  pc: Precomputed parameters. Call ftvq with this
%      and observed image.
function pc = fpc(k1,k2,k3,sz)

pad = (length(k1)-1)/2 * 3;

sz = sz(1:2); sz = sz + 2*pad;

den1 = abs(psf2otf([-1 1],sz)).^2 / 2 + abs(psf2otf([-1 1]',sz)).^2 / 2;

k1f = psf2otf(k1,sz);
k2f = psf2otf(k2,sz);
k3f = psf2otf(k3,sz);

den2 = abs(k1f).^2;
den2(:,:,2) = abs(k2f).^2;
den2(:,:,3) = abs(k3f).^2;


pc = {pad, k1f, k2f, k3f, den1, den2};
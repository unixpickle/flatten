(function () {

    const nn = self.nn;

    class GaussianDiffusion {
        constructor(timestepMap, betas) {
            this.timestepMap = timestepMap;
            this.betas = betas;

            this.alphasCumprod = [];
            this.alphasCumprodPrev = [];
            this.sqrtRecipAlphasCumprod = [];
            this.sqrtRecipm1AlphasCumprod = [];
            let cumprod = 1.0;
            betas.forEach((beta) => {
                const alpha = 1 - beta;
                this.alphasCumprodPrev.push(cumprod);
                cumprod *= alpha;
                this.alphasCumprod.push(cumprod);
                this.sqrtRecipAlphasCumprod.push(Math.sqrt(1 / cumprod));
                this.sqrtRecipm1AlphasCumprod.push(Math.sqrt(1 / cumprod - 1));
            });
        }

        static linearDiffusion128() {
            return new GaussianDiffusion(TIMESTEP_MAP_128, LINEAR_BETAS_128);
        }

        ddimStep(model, x, t, cond) {
            const ts = nn.Tensor.fromData([this.timestepMap[t]]).repeat(0, x.shape[0]);
            const fullOut = model.forward(x, ts, cond);
            const eps = fullOut.slice(1, 0, fullOut.shape[1] / 2);
            const predX0 = x.scale(this.sqrtRecipAlphasCumprod[t])
                .sub(eps.scale(this.sqrtRecipm1AlphasCumprod[t]));
            const alphaBar = this.alphasCumprod[t];
            const alphaBarPrev = this.alphasCumprodPrev[t];
            return predX0
                .scale(Math.sqrt(alphaBarPrev))
                .add(eps.scale(Math.sqrt(1 - alphaBarPrev)));
        }

        ddimSample(model, x0, cond) {
            let x = x0;
            for (let t = this.alphasCumprod.length - 1; t >= 0; --t) {
                x = this.ddimStep(model, x, t, cond);
            }
            return x;
        }
    }

    nn.GaussianDiffusion = GaussianDiffusion;

    const LINEAR_BETAS_128 = [
        9.76562500e-05, 1.46419891e-03, 2.67778225e-03, 3.89007483e-03,
        5.10107781e-03, 6.31079237e-03, 7.51921968e-03, 8.72636093e-03,
        9.93221728e-03, 1.11367899e-02, 1.39562734e-02, 1.36922497e-02,
        1.48928182e-02, 1.60921079e-02, 1.72901198e-02, 1.84868551e-02,
        1.96823151e-02, 2.08765008e-02, 2.20694135e-02, 2.32610542e-02,
        2.44514242e-02, 2.56405246e-02, 2.68283567e-02, 2.80149214e-02,
        2.92002201e-02, 3.03842538e-02, 3.15670237e-02, 3.27485310e-02,
        3.81707026e-02, 3.52550470e-02, 3.64326160e-02, 3.76089272e-02,
        3.87839816e-02, 3.99577805e-02, 4.11303250e-02, 4.23016162e-02,
        4.34716553e-02, 4.46404434e-02, 4.58079818e-02, 4.69742714e-02,
        4.81393135e-02, 4.93031092e-02, 5.04656597e-02, 5.16269660e-02,
        5.27870294e-02, 5.39458510e-02, 6.18551156e-02, 5.64042288e-02,
        5.75591771e-02, 5.87128882e-02, 5.98653633e-02, 6.10166036e-02,
        6.21666101e-02, 6.33153841e-02, 6.44629266e-02, 6.56092388e-02,
        6.67543218e-02, 6.78981767e-02, 6.90408047e-02, 7.01822069e-02,
        7.13223845e-02, 7.24613384e-02, 7.35990700e-02, 7.47355802e-02,
        8.50196915e-02, 7.71466146e-02, 7.82793155e-02, 7.94107997e-02,
        8.05410684e-02, 8.16701227e-02, 8.27979637e-02, 8.39245925e-02,
        8.50500102e-02, 8.61742179e-02, 8.72972168e-02, 8.84190080e-02,
        8.95395925e-02, 9.06589716e-02, 9.17771462e-02, 9.28941176e-02,
        9.40098868e-02, 9.51244549e-02, 1.07674441e-01, 9.74889294e-02,
        9.85997513e-02, 9.97093768e-02, 1.00817807e-01, 1.01925043e-01,
        1.03031085e-01, 1.04135936e-01, 1.05239595e-01, 1.06342065e-01,
        1.07443346e-01, 1.08543439e-01, 1.09642346e-01, 1.10740067e-01,
        1.11836603e-01, 1.12931957e-01, 1.14026128e-01, 1.15119118e-01,
        1.29829210e-01, 1.17437806e-01, 1.18527112e-01, 1.19615242e-01,
        1.20702196e-01, 1.21787975e-01, 1.22872582e-01, 1.23956016e-01,
        1.25038279e-01, 1.26119372e-01, 1.27199295e-01, 1.28278051e-01,
        1.29355641e-01, 1.30432064e-01, 1.31507323e-01, 1.32581418e-01,
        1.33654350e-01, 1.34726122e-01, 1.51493680e-01, 1.36999784e-01,
        1.38067933e-01, 1.39134924e-01, 1.40200760e-01, 1.41265441e-01,
        1.42328968e-01, 1.43391343e-01, 1.44452566e-01, 1.45512638e-01
    ];
    const TIMESTEP_MAP_128 = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 81,
        89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185,
        193, 201, 209, 217, 226, 234, 242, 250, 258, 266, 274, 282, 290,
        298, 306, 314, 322, 330, 338, 346, 354, 362, 371, 379, 387, 395,
        403, 411, 419, 427, 435, 443, 451, 459, 467, 475, 483, 491, 499,
        507, 516, 524, 532, 540, 548, 556, 564, 572, 580, 588, 596, 604,
        612, 620, 628, 636, 644, 652, 661, 669, 677, 685, 693, 701, 709,
        717, 725, 733, 741, 749, 757, 765, 773, 781, 789, 797, 806, 814,
        822, 830, 838, 846, 854, 862, 870, 878, 886, 894, 902, 910, 918,
        926, 934, 942, 951, 959, 967, 975, 983, 991, 999, 1007, 1015, 1023,
    ];

})();
(function () {

    const nn = self.nn;

    class DiffusionModel {
        constructor(rawParams) {
            const params = transposeWeights(rawParams);
            this.params = params;
            this.dModel = this.params["time_embed.2.bias"].shape[0];
            this.dCond = this.params["cond_embed.0.weight"].shape[1];
            this.timeEmbed = new nn.Sequential([
                new nn.Linear(this.params["time_embed.0.weight"], this.params["time_embed.0.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["time_embed.2.weight"], this.params["time_embed.2.bias"]),
            ]);
            this.condEmbed = new nn.Sequential([
                new nn.Linear(this.params["cond_embed.0.weight"], this.params["cond_embed.0.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["cond_embed.2.weight"], this.params["cond_embed.2.bias"]),
            ]);
            this.inputEmbed = new nn.Sequential([
                new nn.Linear(this.params["input_embed.0.weight"], this.params["input_embed.0.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["input_embed.2.weight"], this.params["input_embed.2.bias"]),
            ]);
            this.backbone = new nn.Sequential([
                new nn.Linear(this.params["backbone.0.weight"], this.params["backbone.0.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["backbone.2.weight"], this.params["backbone.2.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["backbone.4.weight"], this.params["backbone.4.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["backbone.6.weight"], this.params["backbone.6.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["backbone.8.weight"], this.params["backbone.8.bias"]),
                new nn.ReLU(),
                new nn.Linear(this.params["backbone.10.weight"], this.params["backbone.10.bias"]),
            ]);
        }

        static async load(path) {
            const data = await (await fetch(path)).json();
            const params = {};
            Object.keys(data).forEach((k) => params[k] = nn.Tensor.fromData(data[k]));
            return new DiffusionModel(params);
        }

        static zeros() {
            return new DiffusionModel({
                "time_embed.0.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "time_embed.0.bias": nn.Tensor.zeros(Shape.from(512)),
                "time_embed.2.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "time_embed.2.bias": nn.Tensor.zeros(Shape.from(512)),
                "cond_embed.0.weight": nn.Tensor.zeros(Shape.from(512, 8)),
                "cond_embed.0.bias": nn.Tensor.zeros(Shape.from(512)),
                "cond_embed.2.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "cond_embed.2.bias": nn.Tensor.zeros(Shape.from(512)),
                "input_embed.0.weight": nn.Tensor.zeros(Shape.from(512, 13)),
                "input_embed.0.bias": nn.Tensor.zeros(Shape.from(512)),
                "input_embed.2.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "input_embed.2.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.0.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.0.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.2.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.2.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.4.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.4.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.6.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.6.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.8.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.8.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.10.weight": nn.Tensor.zeros(Shape.from(26, 512)),
                "backbone.10.bias": nn.Tensor.zeros(Shape.from(26)),
            });
        }

        forward(x, t, cond) {
            const timeEmb = this.timeEmbed.forward(timestepEmbedding(t, this.dModel));
            const inputEmb = this.inputEmbed.forward(x);
            const condEmb = this.condEmbed.forward(cond);
            const combined = timeEmb.add(inputEmb).add(condEmb).scale(1 / Math.sqrt(3));
            return this.backbone.forward(combined);
        }
    }

    function timestepEmbedding(timesteps, dim) {
        const maxPeriod = 10000;
        const range = nn.Tensor.zeros(nn.Shape.make(1, dim / 2));
        for (let i = 0; i < dim / 2; ++i) {
            range.data[i] = i;
        }
        const freqs = (range.scale(-Math.log(maxPeriod) / (dim / 2)))
            .exp()
            .repeat(0, timesteps.shape[0]);
        const args = freqs.mul(timesteps.reshape(nn.Shape.make(-1, 1)).repeat(1, freqs.shape[1]));
        const embedding = nn.Tensor.cat([args.cos(), args.sin()], 1);
        return embedding;
    }

    class StretchModel {
        constructor(rawParams) {
            const params = transposeWeights(rawParams);
            this.params = params;
            this.backbone = new nn.Sequential([
                new nn.Conv2d(params['layers.0.weight'], params['layers.0.bias'], 2),
                new nn.ReLU(),
                new nn.Conv2d(params['layers.2.weight'], params['layers.2.bias'], 2),
                new nn.ReLU(),
                new nn.Conv2d(params['layers.4.weight'], params['layers.4.bias'], 2),
                new nn.ReLU(),
                new nn.Conv2d(params['layers.6.weight'], params['layers.6.bias']),
                new nn.ReLU(),
                new nn.AvgAndFlatten(),
                new nn.Linear(params['layers.10.weight'], params['layers.10.bias']),
            ]);
            this.ratios = params['ratios'];
        }

        static async load(path) {
            const data = await (await fetch(path)).json();
            const params = {};
            Object.keys(data).forEach((k) => params[k] = nn.Tensor.fromData(data[k]));
            return new StretchModel(params);
        }

        forward(x) {
            return this.backbone.forward(x);
        }

        predict(x) {
            const logits = this.forward(x);
            const results = [];
            let offset = 0;
            for (let i = 0; i < logits.shape[0]; ++i) {
                let maxIndex = 0;
                let maxValue = logits.data[offset];
                for (let j = 0; j < logits.shape[1]; ++j) {
                    const x = logits.data[offset++];
                    if (x > maxValue) {
                        maxValue = x;
                        maxIndex = j;
                    }
                }
                results.push(this.ratios.data[maxIndex]);
            }
            return nn.Tensor.fromData([results]);
        }
    }

    function transposeWeights(rawParams) {
        const params = {};
        Object.keys(rawParams).forEach((k) => {
            let v = rawParams[k];
            if (v.shape.length === 2) {
                v = v.t();
            }
            params[k] = v;
        });
        return params
    }

    nn.DiffusionModel = DiffusionModel;
    nn.timestepEmbedding = timestepEmbedding;
    nn.StretchModel = StretchModel;

})();
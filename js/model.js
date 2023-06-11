(function () {

    const nn = window.nn;

    class DiffusionModel {
        constructor(params) {
            this.params = params;
            this.dModel = this.params["time_embed.2.bias"].shape[0];
            this.dCond = this.condEmbed.params["cond_embed.0.weight"].shape[1];
            this.timeEmbed = nn.Sequential([
                nn.Linear(this.params["time_embed.0.weight"], this.params["time_embed.0.bias"]),
                nn.ReLU(),
                nn.Linear(this.params["time_embed.2.weight"], this.params["time_embed.2.bias"]),
            ]);
            this.condEmbed = nn.Sequential([
                nn.Linear(this.params["cond_embed.0.weight"], this.params["cond_embed.0.bias"]),
                nn.ReLU(),
                nn.Linear(this.params["cond_embed.2.weight"], this.params["cond_embed.2.bias"]),
            ]);
            this.inputEmbed = nn.Sequential([
                nn.Linear(this.params["input_embed.0.weight"], this.params["input_embed.0.bias"]),
                nn.ReLU(),
                nn.Linear(this.params["input_embed.2.weight"], this.params["input_embed.2.bias"]),
            ]);
            this.backbone = nn.Sequential([
                nn.Linear(this.params["backbone.0.weight"], this.params["backbone.0.bias"]),
                nn.ReLU(),
                nn.Linear(this.params["backbone.2.weight"], this.params["backbone.2.bias"]),
                nn.ReLU(),
                nn.Linear(this.params["backbone.4.weight"], this.params["backbone.4.bias"]),
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
                "input_embed.0.weight": nn.Tensor.zeros(Shape.from(512, 11)),
                "input_embed.0.bias": nn.Tensor.zeros(Shape.from(512)),
                "input_embed.2.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "input_embed.2.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.0.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.0.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.2.weight": nn.Tensor.zeros(Shape.from(512, 512)),
                "backbone.2.bias": nn.Tensor.zeros(Shape.from(512)),
                "backbone.4.weight": nn.Tensor.zeros(Shape.from(22, 512)),
                "backbone.4.bias": nn.Tensor.zeros(Shape.from(22)),
            });
        }

        forward(x, t, cond) {
            const timeEmb = this.timeEmbed.forward(timestepEmbedding(t, this.dModel));
            const inputEmb = this.inputEmbed(x);
            const condEmb = this.condEmbed(cond);
            const combined = timeEmb.add(inputEmb).add(condEmb).scale(1 / Math.sqrt(3));
            return this.backbone(combined);
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

    nn.DiffusionModel = DiffusionModel;
    nn.timestepEmbedding = timestepEmbedding;

})();
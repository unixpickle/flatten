(function () {

    const nn = window.nn;

    class DiffusionModel {
        constructor(params) {
            this.params = params;
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
    }

    nn.DiffusionModel = DiffusionModel;

})();
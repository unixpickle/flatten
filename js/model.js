var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class DiffusionModel {
    constructor(rawParams, numFeats) {
        this.numFeats = numFeats;
        const params = transposeWeights(rawParams);
        this.numFeats = numFeats || 30;
        this.params = params;
        this.dModel = this.params["time_embed.2.bias"].shape[0];
        this.dCond = this.params["cond_embed.0.weight"].shape[1] / (1 + this.numFeats);
        this.timeEmbed = new Sequential([
            new Linear(this.params["time_embed.0.weight"], this.params["time_embed.0.bias"]),
            new ReLU(),
            new Linear(this.params["time_embed.2.weight"], this.params["time_embed.2.bias"]),
        ]);
        this.condEmbed = new Sequential([
            new Linear(this.params["cond_embed.0.weight"], this.params["cond_embed.0.bias"]),
            new ReLU(),
            new Linear(this.params["cond_embed.2.weight"], this.params["cond_embed.2.bias"]),
        ]);
        this.inputEmbed = new Sequential([
            new Linear(this.params["input_embed.0.weight"], this.params["input_embed.0.bias"]),
            new ReLU(),
            new Linear(this.params["input_embed.2.weight"], this.params["input_embed.2.bias"]),
        ]);
        this.backbone = new Sequential([
            new Linear(this.params["backbone.0.weight"], this.params["backbone.0.bias"]),
            new ReLU(),
            new Linear(this.params["backbone.2.weight"], this.params["backbone.2.bias"]),
            new ReLU(),
            new Linear(this.params["backbone.4.weight"], this.params["backbone.4.bias"]),
            new ReLU(),
            new Linear(this.params["backbone.6.weight"], this.params["backbone.6.bias"]),
            new ReLU(),
            new Linear(this.params["backbone.8.weight"], this.params["backbone.8.bias"]),
            new ReLU(),
            new Linear(this.params["backbone.10.weight"], this.params["backbone.10.bias"]),
        ]);
    }
    static load(path) {
        return __awaiter(this, void 0, void 0, function* () {
            return new DiffusionModel(yield readParamDict(path));
        });
    }
    forward(x, t, cond) {
        const timeEmb = this.timeEmbed.forward(timestepEmbedding(t, this.dModel));
        const inputEmb = this.inputEmbed.forward(x);
        const condEmb = this.condEmbed.forward(frequencyPosEmbedding(cond, this.numFeats));
        const combined = timeEmb.add(inputEmb).add(condEmb).scale(1 / Math.sqrt(3));
        return this.backbone.forward(combined);
    }
}
function timestepEmbedding(timesteps, dim) {
    const maxPeriod = 10000;
    const range = Tensor.zeros(Shape.make(1, dim / 2));
    for (let i = 0; i < dim / 2; ++i) {
        range.data[i] = i;
    }
    const freqs = (range.scale(-Math.log(maxPeriod) / (dim / 2)))
        .exp()
        .repeat(0, timesteps.shape[0]);
    const args = freqs.mul(timesteps.reshape(Shape.make(-1, 1)).repeat(1, freqs.shape[1]));
    const embedding = Tensor.cat([args.cos(), args.sin()], 1);
    return embedding;
}
function frequencyPosEmbedding(coords, numFeats) {
    const maxArg = 1000.0;
    if (!numFeats) {
        return coords;
    }
    const coeffs = Tensor.zeros(Shape.make(numFeats / 2));
    const maxLog = Math.log(maxArg);
    for (let i = 0; i < numFeats / 2; i++) {
        coeffs.data[i] = Math.exp(i * (maxLog / (numFeats / 2 - 1)));
    }
    const repCoeffs = coeffs.unsqueeze(0).unsqueeze(0).repeat(0, coords.shape[0]).repeat(1, coords.shape[1]);
    let args = coords.unsqueeze(-1).repeat(coords.shape.length, numFeats / 2).mul(repCoeffs);
    args = args.reshape(Shape.make(args.shape[0], -1));
    return Tensor.cat([coords, args.cos(), args.sin()], 1);
}
class StretchModel {
    constructor(rawParams) {
        const params = transposeWeights(rawParams);
        this.params = params;
        this.backbone = new Sequential([
            new Conv2d(params['layers.0.weight'], params['layers.0.bias'], 2),
            new ReLU(),
            new Conv2d(params['layers.2.weight'], params['layers.2.bias'], 2),
            new ReLU(),
            new Conv2d(params['layers.4.weight'], params['layers.4.bias'], 2),
            new ReLU(),
            new Conv2d(params['layers.6.weight'], params['layers.6.bias']),
            new ReLU(),
            new AvgAndFlatten(),
            new Linear(params['layers.10.weight'], params['layers.10.bias']),
        ]);
        this.ratios = params['ratios'];
    }
    static load(path) {
        return __awaiter(this, void 0, void 0, function* () {
            return new StretchModel(yield readParamDict(path));
        });
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
        return Tensor.fromData([results]);
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
    return params;
}
function readParamDict(url) {
    return __awaiter(this, void 0, void 0, function* () {
        const buf = yield (yield fetch(url)).arrayBuffer();
        const bytes = new Uint8Array(buf);
        const metadataSize = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
        const metadata = JSON.parse(String.fromCharCode.apply(null, bytes.slice(4, 4 + metadataSize)));
        let allData = new Float32Array(flipToLittleEndian(buf.slice(4 + metadataSize)));
        const stateDict = {};
        metadata.forEach((info) => {
            const [name, rawShape] = info;
            const shape = Shape.make(...rawShape);
            const param = new Tensor(allData.slice(0, shape.numel()), shape, null);
            allData = allData.slice(shape.numel());
            stateDict[name] = param;
        });
        return stateDict;
    });
}
function flipToLittleEndian(input) {
    if (!isBigEndian()) {
        return input;
    }
    let arr = new Uint8Array(input);
    const output = new ArrayBuffer(arr.length);
    const out = new Uint8Array(output);
    for (let i = 0; i < arr.length; i += 4) {
        const w = arr[i];
        const x = arr[i + 1];
        const y = arr[i + 2];
        const z = arr[i + 3];
        out[i] = z;
        out[i + 1] = y;
        out[i + 2] = x;
        out[i + 3] = w;
    }
    return output;
}
function isBigEndian() {
    const x = new ArrayBuffer(4);
    new Float32Array(x)[0] = 1;
    return new Uint8Array(x)[0] != 0;
}
//# sourceMappingURL=model.js.map
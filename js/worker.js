var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const ctx = self;
importScripts("nn.js", "model.js", "diffusion.js", "solver.js", "pixel_source.js");
const ITERATIONS = 10000;
const STEP_SIZE = 0.001;
let diffusion = GaussianDiffusion.linearDiffusion32();
let diffusionModel = null;
let stretchModel = null;
onmessage = (event) => {
    const methods = {
        "solve": solve,
        "predictStretch": predictStretch,
        "exportImage": exportImage,
    };
    const msg = event.data;
    if (!methods.hasOwnProperty(msg.method)) {
        postMessage({ id: msg.id, error: "no such method: " + msg.method });
        return;
    }
    const statusFn = (x) => {
        postMessage({ id: msg.id, status: x });
    };
    methods[msg.method].apply(null, msg.args.concat([statusFn])).then((x) => {
        postMessage({ id: msg.id, data: x });
    }).catch((e) => {
        postMessage({ id: msg.id, error: "" + e });
    });
};
function solve(cornerData, statusFn) {
    return __awaiter(this, void 0, void 0, function* () {
        const corners = Tensor.fromData(cornerData);
        const attempts = 4;
        statusFn("Loading diffusion model...");
        const model = yield getDiffusionModel();
        statusFn("Sampling diffusion model...");
        const samples = diffusion.ddimSample(model, Tensor.randn(Shape.make(attempts, 13)), corners.reshape(Shape.make(1, -1)).repeat(0, attempts));
        let bestLoss = null;
        let bestSolution = null;
        for (let i = 0; i < attempts; ++i) {
            statusFn("Refining solution " + (i + 1) + "/" + attempts + "...");
            const row = samples.slice(0, i, i + 1).reshape(Shape.make(-1));
            const solution = PerspectiveSolution.fromFlatVec(row);
            const [initLoss, finalLoss] = solution.iterate(corners, ITERATIONS, STEP_SIZE);
            if (bestLoss === null || finalLoss < bestLoss) {
                bestLoss = finalLoss;
                bestSolution = solution;
            }
        }
        return bestSolution.toFlatVec().toList();
    });
}
function predictStretch(imageData, statusFn) {
    return __awaiter(this, void 0, void 0, function* () {
        const x = Tensor.fromData(imageData);
        statusFn("Loading aspect ratio model...");
        const model = yield getStretchModel();
        statusFn("Predicting aspect ratio...");
        const pred = model.predict(x);
        return pred.toList();
    });
}
function exportImage(rawSolution, rawPixelSource, aspectRatio, sideLength, _) {
    return __awaiter(this, void 0, void 0, function* () {
        const solution = PerspectiveSolution.fromFlatVec(Tensor.fromData(rawSolution));
        const pixelSource = PixelSource.deserialize(rawPixelSource);
        const w = 1;
        const h = aspectRatio;
        const scale = Math.min(sideLength / w, sideLength / h);
        const dstCanvas = new OffscreenCanvas(w * scale, h * scale);
        pixelSource.extractProjectedImage(solution, dstCanvas);
        const blob = yield dstCanvas.convertToBlob({ type: "image/png" });
        const fr = new FileReaderSync();
        let dt = fr.readAsDataURL(blob);
        dt = dt.replace(/^data:image\/[^;]*/, "data:application/octet-stream");
        const filename = "flattened_" + Math.round(new Date().getTime() / 1000) + ".png";
        dt = dt.replace(/^data:application\/octet-stream/, "data:application/octet-stream;headers=Content-Disposition%3A%20attachment%3B%20filename=" + filename);
        return dt;
    });
}
function getDiffusionModel() {
    return __awaiter(this, void 0, void 0, function* () {
        if (diffusionModel !== null) {
            return diffusionModel;
        }
        diffusionModel = yield DiffusionModel.load("../models/diffusion.bin");
        return diffusionModel;
    });
}
function getStretchModel() {
    return __awaiter(this, void 0, void 0, function* () {
        if (stretchModel !== null) {
            return stretchModel;
        }
        stretchModel = yield StretchModel.load("../models/stretch.bin");
        return stretchModel;
    });
}
//# sourceMappingURL=worker.js.map
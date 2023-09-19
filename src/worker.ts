/// <reference lib="webworker" />

const ctx: WorkerGlobalScope = (self as any)

importScripts(
    "nn.js",
    "model.js",
    "diffusion.js",
    "solver.js",
    "pixel_source.js",
);

const ITERATIONS = 10000;
const STEP_SIZE = 0.001;

let diffusion = GaussianDiffusion.linearDiffusion32();
let diffusionModel: DiffusionModel = null;
let stretchModel: StretchModel = null;

onmessage = (event) => {
    const methods: { [key: string]: (...args: any[]) => Promise<any> } = {
        "solve": solve,
        "predictStretch": predictStretch,
        "exportImage": exportImage,
    }
    const msg = event.data;
    if (!methods.hasOwnProperty(msg.method)) {
        postMessage({ id: msg.id, error: "no such method: " + msg.method });
        return;
    }
    const statusFn = (x: string) => {
        postMessage({ id: msg.id, status: x });
    };
    methods[msg.method].apply(null, msg.args.concat([statusFn])).then((x: any) => {
        postMessage({ id: msg.id, data: x });
    }).catch((e: any) => {
        postMessage({ id: msg.id, error: "" + e });
    });
}

async function solve(cornerData: TensorNativeData, statusFn: StatusFunc) {
    const corners = Tensor.fromData(cornerData);
    const attempts = 4;
    statusFn("Loading diffusion model...");
    const model = await getDiffusionModel();
    statusFn("Sampling diffusion model...");
    const samples = diffusion.ddimSample(
        model,
        Tensor.randn(Shape.make(attempts, 13)),
        corners.reshape(Shape.make(1, -1)).repeat(0, attempts),
    );
    let bestLoss = null;
    let bestSolution = null;
    for (let i = 0; i < attempts; ++i) {
        statusFn("Refining solution " + (i + 1) + "/" + attempts + "...");
        const row = samples.slice(0, i, i + 1).reshape(Shape.make(-1));
        const solution = PerspectiveSolution.fromFlatVec(row);
        const [initLoss, finalLoss] = solution.iterate(corners, ITERATIONS, STEP_SIZE);
        // console.log("solution " + i + ": loss went from " + initLoss + " => " + finalLoss);
        if (bestLoss === null || finalLoss < bestLoss) {
            bestLoss = finalLoss;
            bestSolution = solution;
        }
    }
    return bestSolution.toFlatVec().toList();
}

async function predictStretch(imageData: TensorNativeData, statusFn: StatusFunc) {
    const x = Tensor.fromData(imageData);
    statusFn("Loading aspect ratio model...");
    const model = await getStretchModel();
    statusFn("Predicting aspect ratio...");
    const pred = model.predict(x);
    return pred.toList();
}

async function exportImage(
    rawSolution: TensorNativeData,
    rawPixelSource: SerializedPixelSource,
    aspectRatio: number,
    sideLength: number,
    _: StatusFunc,
): Promise<string> {
    const solution = PerspectiveSolution.fromFlatVec(Tensor.fromData(rawSolution));
    const pixelSource = PixelSource.deserialize(rawPixelSource);

    const w = 1;
    const h = aspectRatio;
    const scale = Math.min(sideLength / w, sideLength / h);

    const dstCanvas = new OffscreenCanvas(w * scale, h * scale);
    pixelSource.extractProjectedImage(solution, dstCanvas);

    // https://stackoverflow.com/questions/12796513/html5-canvas-to-png-file
    const blob = await dstCanvas.convertToBlob({ type: "image/png" });
    const fr = new FileReaderSync();
    let dt = fr.readAsDataURL(blob);
    dt = dt.replace(/^data:image\/[^;]*/, "data:application/octet-stream");

    const filename = "flattened_" + Math.round(new Date().getTime() / 1000) + ".png";
    dt = dt.replace(
        /^data:application\/octet-stream/,
        "data:application/octet-stream;headers=Content-Disposition%3A%20attachment%3B%20filename=" + filename,
    );

    return dt;
}

async function getDiffusionModel() {
    if (diffusionModel !== null) {
        return diffusionModel;
    }
    diffusionModel = await DiffusionModel.load("../models/diffusion.bin");
    return diffusionModel;
}

async function getStretchModel() {
    if (stretchModel !== null) {
        return stretchModel;
    }
    stretchModel = await StretchModel.load("../models/stretch.bin");
    return stretchModel;
}

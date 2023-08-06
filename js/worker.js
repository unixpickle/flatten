importScripts(
    "nn.js",
    "model.js",
    "diffusion.js",
    "solver.js",
    "pixel_source.js",
);

const ITERATIONS = 10000;
const STEP_SIZE = 0.001;

let diffusion = nn.GaussianDiffusion.linearDiffusion32();
let diffusionModel = null;
let stretchModel = null;

onmessage = (event) => {
    const methods = {
        "solve": solve,
        "predictStretch": predictStretch,
        "exportImage": exportImage,
    }
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
}

async function solve(cornerData, statusFn) {
    const corners = nn.Tensor.fromData(cornerData);
    const attempts = 4;
    statusFn("Loading diffusion model...");
    const model = await getDiffusionModel();
    statusFn("Sampling diffusion model...");
    const samples = diffusion.ddimSample(
        model,
        nn.Tensor.randn(nn.Shape.make(attempts, 13)),
        corners.reshape(nn.Shape.make(1, -1)).repeat(0, attempts),
    );
    let bestLoss = null;
    let bestSolution = null;
    for (let i = 0; i < attempts; ++i) {
        statusFn("Refining solution " + (i + 1) + "/" + attempts + "...");
        const row = samples.slice(0, i, i + 1).reshape(nn.Shape.make(-1));
        const solution = nn.PerspectiveSolution.fromFlatVec(row);
        const [initLoss, finalLoss] = solution.iterate(corners, ITERATIONS, STEP_SIZE);
        // console.log("solution " + i + ": loss went from " + initLoss + " => " + finalLoss);
        if (bestLoss === null || finalLoss < bestLoss) {
            bestLoss = finalLoss;
            bestSolution = solution;
        }
    }
    return bestSolution.toFlatVec().toList();
}

async function predictStretch(imageData, statusFn) {
    const x = nn.Tensor.fromData(imageData);
    statusFn("Loading aspect ratio model...");
    const model = await getStretchModel();
    statusFn("Predicting aspect ratio...");
    const pred = model.predict(x);
    return pred.toList();
}

async function exportImage(rawSolution, rawPixelSource, aspectRatio, sideLength, statusFn) {
    const solution = nn.PerspectiveSolution.fromFlatVec(nn.Tensor.fromData(rawSolution));
    const pixelSource = nn.PixelSource.deserialize(rawPixelSource);

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
    diffusionModel = await nn.DiffusionModel.load("../models/diffusion.bin");
    return diffusionModel;
}

async function getStretchModel() {
    if (stretchModel !== null) {
        return stretchModel;
    }
    stretchModel = await nn.StretchModel.load("../models/stretch.bin");
    return stretchModel;
}

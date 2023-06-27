importScripts(
    'nn.js',
    'model.js',
    'diffusion.js',
    'solver.js',
);

const ITERATIONS = 10000;
const STEP_SIZE = 0.001;

let diffusion = nn.GaussianDiffusion.linearDiffusion128();
let diffusionModel = null;

onmessage = (event) => {
    const cornerData = event.data.points;
    solve(cornerData).then((solution) => {
        postMessage({ solution: solution });
    }).catch((e) => {
        postMessage({ error: e });
    });
}

async function solve(cornerData) {
    const corners = nn.Tensor.fromData(cornerData);
    const attempts = 4;
    const samples = diffusion.ddimSample(
        await getDiffusionModel(),
        nn.Tensor.randn(nn.Shape.make(attempts, 11)),
        corners.reshape(nn.Shape.make(1, -1)).repeat(0, attempts),
    );
    let bestLoss = null;
    let bestSolution = null;
    for (let i = 0; i < attempts; ++i) {
        const row = samples.slice(0, i, i + 1).reshape(nn.Shape.make(-1));
        const solution = nn.PerspectiveSolution.fromFlatVec(row);
        const [initLoss, finalLoss] = solution.iterate(corners, ITERATIONS, STEP_SIZE);
        console.log(initLoss, finalLoss);
        if (bestLoss === null || finalLoss < bestLoss) {
            bestLoss = finalLoss;
            bestSolution = solution;
        }
    }
    return bestSolution.toFlatVec().toList();
}

async function getDiffusionModel() {
    if (diffusionModel !== null) {
        return diffusionModel;
    }
    diffusionModel = await nn.DiffusionModel.load('../models/diffusion.json');
    return diffusionModel;
}

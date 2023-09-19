type PerspectiveSolutionField = "origin" | "size" | "rotation" | "translation" | "postTranslation";
const PerspectiveSolutionFields: PerspectiveSolutionField[] = [
    "origin", "size", "rotation", "translation", "postTranslation",
];

class PerspectiveSolution {
    constructor(
        public origin: Tensor,
        public size: Tensor,
        public rotation: Tensor,
        public translation: Tensor,
        public postTranslation: Tensor,
    ) {
    }

    static fromFlatVec(sample: Tensor): PerspectiveSolution {
        if (!sample.shape.equals(Shape.make(13))) {
            throw new Error("unexpected shape for sample: " + sample.shape);
        }
        return new PerspectiveSolution(
            sample.slice(0, 0, 3),
            sample.slice(0, 3, 5),
            sample.slice(0, 5, 8),
            sample.slice(0, 8, 11),
            sample.slice(0, 11, 13),
        );
    }

    static zeros(): PerspectiveSolution {
        return new PerspectiveSolution(
            Tensor.zeros(Shape.make(3)),
            Tensor.zeros(Shape.make(2)),
            Tensor.zeros(Shape.make(3)),
            Tensor.zeros(Shape.make(3)),
            Tensor.zeros(Shape.make(2)),
        );
    }

    toFlatVec(): Tensor {
        return Tensor.cat([
            this.origin, this.size, this.rotation, this.translation, this.postTranslation
        ], 0);
    }

    iterate(corners: Tensor, numIters: number, stepSize: number) {
        let finalLoss = null;
        let initLoss = null;
        const opt = new AdamOptimizer(stepSize);
        for (let i = 0; i < numIters; ++i) {
            const [loss, grad] = this.lossAndGrad((x) => x.projectionMSE(corners));
            opt.update(this, grad);
            finalLoss = loss;
            if (i === 0) {
                initLoss = loss;
            }
        }
        return [initLoss, finalLoss];
    }

    addScaled(gradient: PerspectiveSolution, scale: number) {
        return new PerspectiveSolution(
            this.origin.add(gradient.origin.scale(scale)),
            this.size.add(gradient.size.scale(scale)),
            this.rotation.add(gradient.rotation.scale(scale)),
            this.translation.add(gradient.translation.scale(scale)),
            this.postTranslation.add(gradient.postTranslation.scale(scale)),
        );
    }

    lossAndGrad(lossFn: (_: PerspectiveSolution) => Tensor): [number, PerspectiveSolution] {
        const gSolution = new PerspectiveSolution(
            this.origin.detach(),
            this.size.detach(),
            this.rotation.detach(),
            this.translation.detach(),
            this.postTranslation.detach(),
        );
        const grad = new PerspectiveSolution(null, null, null, null, null);
        gSolution.origin.backward = (x) => grad.origin = x;
        gSolution.size.backward = (x) => grad.size = x;
        gSolution.rotation.backward = (x) => grad.rotation = x;
        gSolution.translation.backward = (x) => grad.translation = x;
        gSolution.postTranslation.backward = (x) => grad.postTranslation = x;
        const loss = lossFn(gSolution);
        loss.backward(Tensor.fromData(1));
        return [loss.data[0], grad];
    }

    projector(): (_: Tensor) => Tensor {
        const rot = this.rotationMatrix();
        const [ox, oy, oz] = this.origin.toList() as number[];
        return (p) => {
            const points3d = Tensor.zeros(Shape.make(p.shape[0], 3));
            for (let i = 0; i < p.shape[0]; i++) {
                points3d.data[i * 3] = p.data[i * 2] + ox;
                points3d.data[i * 3 + 1] = p.data[i * 2 + 1] + oy;
                points3d.data[i * 3 + 2] = oz;
            }
            const proj = cameraProject(
                rot,
                this.translation,
                this.postTranslation,
                points3d,
            );
            return proj;
        };
    }

    projectionMSE(corners: Tensor): Tensor {
        const offsets = this.size.accumGrad((size) => {
            const zero1 = Tensor.fromData([0]);
            const size0 = Tensor.zeros(Shape.make(3));
            const sizeX = Tensor.cat([size.slice(0, 0, 1), Tensor.fromData([0, 0])], 0);
            const sizeY = Tensor.cat([zero1, size.slice(0, 1, 2), zero1], 0);
            const sizeBoth = Tensor.cat([size, zero1], 0);
            return Tensor.cat([
                size0.unsqueeze(0),
                sizeX.unsqueeze(0),
                sizeBoth.unsqueeze(0),
                sizeY.unsqueeze(0),
            ], 0);
        });
        const corners3d = this.origin.unsqueeze(0).repeat(0, 4).add(offsets);
        const rotation = this.rotationMatrix();
        const corners2d = cameraProject(
            rotation,
            this.translation,
            this.postTranslation,
            corners3d,
        );
        return corners.sub(corners2d).pow(2).mean();
    }

    rotationMatrix(): Tensor {
        return this.rotation.accumGrad((angles) => {
            let matrix = null;
            for (let axis = 0; axis < 3; ++axis) {
                const angle = angles.slice(0, axis, axis + 1).reshape(new Shape());
                const mat = rotation(axis, angle);
                if (matrix === null) {
                    matrix = mat;
                } else {
                    matrix = matmul(mat, matrix);
                }
            }
            return matrix;
        });
    }
}

class AdamOptimizer {
    public beta1: number;
    public beta2: number;
    public epsilon: number;

    private moment1: PerspectiveSolution;
    private moment2: PerspectiveSolution;
    private t: number;

    constructor(public lr: number, beta1?: number, beta2?: number, epsilon?: number) {
        this.lr = lr;
        this.beta1 = typeof beta1 === "undefined" ? 0.9 : beta1;
        this.beta2 = typeof beta2 === "undefined" ? 0.999 : beta2;
        this.epsilon = typeof epsilon === "undefined" ? 0.00000001 : epsilon;

        this.moment1 = PerspectiveSolution.zeros();
        this.moment2 = PerspectiveSolution.zeros();
        this.t = 0;
    }

    update(solution: PerspectiveSolution, gradient: PerspectiveSolution) {
        this.t += 1;
        const scale1 = -this.lr / (1 - Math.pow(this.beta1, this.t));
        const scale2 = 1 / (1 - Math.pow(this.beta2, this.t));
        PerspectiveSolutionFields.forEach((k) => {
            this.moment1[k] = (this.moment1[k] as Tensor).scale(this.beta1)
                .add(gradient[k].scale(1 - this.beta1));
            this.moment2[k] = this.moment2[k].scale(this.beta2)
                .add(gradient[k].pow(2).scale(1 - this.beta2));
            const ratio = this.moment1[k].scale(scale1).mul(this.moment2[k].scale(scale2).pow(0.5).addScalar(this.epsilon).pow(-1))
            solution[k] = solution[k].add(ratio);
        });
    }
}

function cameraProject(rotation: Tensor, translation: Tensor, postTranslation: Tensor, points: Tensor): Tensor {
    const rotated = matmul(rotation, points.t()).t();
    const tx = rotated.add(translation.unsqueeze(0).repeat(0, points.shape[0]));
    const perspectivePoint = divideByZ(tx);
    return perspectivePoint.add(postTranslation.unsqueeze(0).repeat(0, points.shape[0]));
}

function divideByZ(coords: Tensor): Tensor {
    if (coords.shape.length !== 2 || coords.shape[1] !== 3) {
        throw new Error("invalid coordinate batch for perspective transform");
    }
    return coords.accumGrad((input) => {
        const xy = input.slice(1, 0, 2);
        const scales = input.slice(1, 2, 3).pow(-1).scale(-1).repeat(1, 2);
        return xy.mul(scales);
    });
}
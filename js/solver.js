(function () {

    const nn = window.nn;

    class PerspectiveSolution {
        constructor(origin, size, rotation, translation) {
            this.origin = origin;
            this.size = size;
            this.rotation = rotation;
            this.translation = translation;
        }

        static fromDiffusionSample(sample) {
            if (!sample.shape.equals(nn.Shape.make(11))) {
                throw new Error("unexpected shape for sample: " + sample.shape);
            }
            return new PerspectiveSolution(
                sample.slice(0, 0, 3),
                sample.slice(0, 3, 5),
                sample.slice(0, 5, 8),
                sample.slice(0, 8, 11),
            );
        }

        iterate(corners, numIters, stepSize) {
            let finalLoss = null;
            let initLoss = null;
            let solution = this;
            for (let i = 0; i < numIters; ++i) {
                const [loss, grad] = solution.lossAndGrad((x) => x.projectionMSE(corners));
                solution = solution.addScaled(grad, -stepSize);
                finalLoss = loss;
                if (i === 0) {
                    initLoss = loss;
                }
            }
            return [solution, initLoss, finalLoss];
        }

        addScaled(gradient, scale) {
            return new PerspectiveSolution(
                this.origin.add(gradient.origin.scale(scale)),
                this.size.add(gradient.size.scale(scale)),
                this.rotation.add(gradient.rotation.scale(scale)),
                this.translation.add(gradient.translation.scale(scale)),
            );
        }

        lossAndGrad(lossFn) {
            const gSolution = new PerspectiveSolution(
                this.origin.detach(),
                this.size.detach(),
                this.rotation.detach(),
                this.translation.detach(),
            );
            const grad = new PerspectiveSolution(null, null, null, null);
            gSolution.origin.backward = (x) => grad.origin = x;
            gSolution.size.backward = (x) => grad.size = x;
            gSolution.rotation.backward = (x) => grad.rotation = x;
            gSolution.translation.backward = (x) => grad.translation = x;
            const loss = lossFn(gSolution);
            loss.backward(nn.Tensor.fromData(1));
            return [loss.data[0], grad];
        }

        projectionMSE(corners) {
            const offsets = this.size.accumGrad((size) => {
                const zero1 = nn.Tensor.fromData([0]);
                const size0 = nn.Tensor.zeros(nn.Shape.make(3));
                const sizeX = nn.Tensor.cat([size.slice(0, 0, 1), nn.Tensor.fromData([0, 0])], 0);
                const sizeY = nn.Tensor.cat([zero1, size.slice(0, 1, 2), zero1], 0);
                const sizeBoth = nn.Tensor.cat([size, zero1], 0);
                return nn.Tensor.cat([
                    size0.unsqueeze(0),
                    sizeX.unsqueeze(0),
                    sizeBoth.unsqueeze(0),
                    sizeY.unsqueeze(0),
                ], 0);
            });
            const corners3d = this.origin.unsqueeze(0).repeat(0, 4).add(offsets);
            const rotation = this.rotation.accumGrad((angles) => {
                let matrix = null;
                for (let axis = 0; axis < 3; ++axis) {
                    const angle = angles.slice(0, axis, axis + 1).reshape(new nn.Shape());
                    const mat = nn.rotation(axis, angle);
                    if (matrix === null) {
                        matrix = mat;
                    } else {
                        matrix = nn.matmul(mat, matrix);
                    }
                }
                return matrix;
            });
            const corners2d = cameraProject(rotation, this.translation, corners3d);
            return corners.sub(corners2d).pow(2).mean();
        }
    }

    function cameraProject(rotation, translation, points) {
        const rotated = nn.matmul(rotation, points.t()).t();
        const tx = rotated.add(translation.unsqueeze(0).repeat(0, points.shape[0]));
        return divideByZ(tx);
    }

    function divideByZ(coords) {
        if (coords.shape.length !== 2 || coords.shape[1] !== 3) {
            throw new Error("invalid coordinate batch for perspective transform");
        }

        // Accumulate input gradients from both ends of the slice.
        // If we did not do this, multiple backward calls could flow
        // to the coords, causing unnecessary computational cost.
        const input = coords.detach();
        let accumGrad = null;
        input.backward = !coords.needsGrad() ? null : (x) => {
            if (!accumGrad) {
                accumGrad = x;
            } else {
                accumGrad = accumGrad.add(x);
            }
        };

        const xy = input.slice(1, 0, 2);
        const scales = input.slice(1, 2, 3).pow(-1).scale(-1).repeat(1, 2);
        const result = xy.mul(scales);

        const oldBackward = result.backward;
        result.backward = !coords.needsGrad() ? null : (g) => {
            accumGrad = null;
            oldBackward.call(result, g);
            coords.backward(accumGrad);
        };

        return result;
    }

    nn.PerspectiveSolution = PerspectiveSolution;

})();
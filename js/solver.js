(function () {

    const nn = self.nn;

    class PerspectiveSolution {
        constructor(origin, size, rotation, translation, postTranslation) {
            this.origin = origin;
            this.size = size;
            this.rotation = rotation;
            this.translation = translation;
            this.postTranslation = postTranslation;
        }

        static fromFlatVec(sample) {
            if (!sample.shape.equals(nn.Shape.make(13))) {
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

        static zeros() {
            return new PerspectiveSolution(
                nn.Tensor.zeros(nn.Shape.make(3)),
                nn.Tensor.zeros(nn.Shape.make(2)),
                nn.Tensor.zeros(nn.Shape.make(3)),
                nn.Tensor.zeros(nn.Shape.make(3)),
                nn.Tensor.zeros(nn.Shape.make(2)),
            );
        }

        toFlatVec() {
            return nn.Tensor.cat([
                this.origin, this.size, this.rotation, this.translation, this.postTranslation
            ], 0);
        }

        iterate(corners, numIters, stepSize) {
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

        addScaled(gradient, scale) {
            return new PerspectiveSolution(
                this.origin.add(gradient.origin.scale(scale)),
                this.size.add(gradient.size.scale(scale)),
                this.rotation.add(gradient.rotation.scale(scale)),
                this.translation.add(gradient.translation.scale(scale)),
                this.postTranslation.add(gradient.postTranslation.scale(scale)),
            );
        }

        lossAndGrad(lossFn) {
            const gSolution = new PerspectiveSolution(
                this.origin.detach(),
                this.size.detach(),
                this.rotation.detach(),
                this.translation.detach(),
                this.postTranslation.detach(),
            );
            const grad = new PerspectiveSolution(null, null, null, null);
            gSolution.origin.backward = (x) => grad.origin = x;
            gSolution.size.backward = (x) => grad.size = x;
            gSolution.rotation.backward = (x) => grad.rotation = x;
            gSolution.translation.backward = (x) => grad.translation = x;
            gSolution.postTranslation.backward = (x) => grad.postTranslation = x;
            const loss = lossFn(gSolution);
            loss.backward(nn.Tensor.fromData(1));
            return [loss.data[0], grad];
        }

        projector() {
            const rot = this.rotationMatrix();
            return (p) => {
                const [ox, oy, oz] = this.origin.toList();
                const point3d = nn.Tensor.fromData([[ox + p.x, oy + p.y, oz]]);
                const proj = cameraProject(
                    rot,
                    this.translation,
                    this.postTranslation,
                    point3d,
                );
                return { x: proj.data[0], y: proj.data[1] };
            };
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
            const rotation = this.rotationMatrix();
            const corners2d = cameraProject(
                rotation,
                this.translation,
                this.postTranslation,
                corners3d,
            );
            return corners.sub(corners2d).pow(2).mean();
        }

        rotationMatrix() {
            return this.rotation.accumGrad((angles) => {
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
        }
    }

    class AdamOptimizer {
        constructor(lr, beta1, beta2, epsilon) {
            this.lr = lr;
            this.beta1 = typeof beta1 === "undefined" ? 0.9 : beta1;
            this.beta2 = typeof beta2 === "undefined" ? 0.999 : beta2;
            this.epsilon = typeof epsilon === "undefined" ? 0.00000001 : epsilon;

            this.moment1 = PerspectiveSolution.zeros();
            this.moment2 = PerspectiveSolution.zeros();
            this.t = 0;
        }

        update(solution, gradient) {
            this.t += 1;
            const scale1 = -this.lr / (1 - Math.pow(this.beta1, this.t));
            const scale2 = 1 / (1 - Math.pow(this.beta2, this.t));
            ['origin', 'size', 'rotation', 'translation'].forEach((k) => {
                this.moment1[k] = this.moment1[k].scale(this.beta1)
                    .add(gradient[k].scale(1 - this.beta1));
                this.moment2[k] = this.moment2[k].scale(this.beta2)
                    .add(gradient[k].pow(2).scale(1 - this.beta2));
                const ratio = this.moment1[k].scale(scale1).mul(this.moment2[k].scale(scale2).pow(0.5).addScalar(this.epsilon).pow(-1))
                solution[k] = solution[k].add(ratio);
            });
        }
    }

    function cameraProject(rotation, translation, postTranslation, points) {
        const rotated = nn.matmul(rotation, points.t()).t();
        const tx = rotated.add(translation.unsqueeze(0).repeat(0, points.shape[0]));
        const perspectivePoint = divideByZ(tx);
        return perspectivePoint.add(postTranslation.unsqueeze(0).repeat(0, points.shape[0]));
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
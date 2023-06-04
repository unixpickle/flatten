(function () {

    class Shape extends Array {
        static make(...args) {
            const res = new Shape();
            args.forEach((x) => res.push(x));
            return res;
        }

        toString() {
            return "Shape(" + this.join(", ") + ")"
        }

        numel() {
            return this.reduce((cur, v) => cur * v, 1);
        }

        infer(numel) {
            const numNeg1 = this.reduce((c, v) => c + (v === -1 ? 1 : 0), 0);

            if (numNeg1 > 1) {
                throw new Error("too many unknown entries in shape: " + this);
            }

            if (numNeg1 === 1) {
                const product = this.reduce((c, v) => c * (v === -1 ? 1 : v));
                if (numel % product !== 0) {
                    throw new Error("incompatible shape " + this + " for " + numel + " elements");
                }
                const res = this.slice();
                res[this.indexOf(-1)] = numel / product;
                return res;
            }

            return this;
        }

        equals(other) {
            return this.length === other.length && !this.some((x, i) => x !== other[i]);
        }
    }

    class Tensor {
        constructor(data, shape, backward) {
            if (data.length !== shape.numel()) {
                throw new Error("data size " + data.length + " does not match shape " + shape);
            }
            this.data = data;
            this.shape = shape;
            this.backward = backward || null;
        }

        static zeros(shape) {
            const data = new Float32Array(shape.numel());
            return new Tensor(data, shape, null);
        }

        static stackOuter(tensors) {
            tensors.forEach((x) => {
                if (!x.shape.equals(tensors[0].shape)) {
                    throw new Error("shape " + x.shape + " does not match first shape " +
                        tensors[0].shape);
                }
            });
            const chunkSize = tensors[0].shape.numel();
            const resultData = new Float32Array(chunkSize * tensors.length);
            tensors.forEach((x, i) => {
                resultData.set(x.data, chunkSize * i);
            });
            const backward = !tensors.some((x) => x.needsGrad()) ? null : (grad) => {
                tensors.forEach((x, i) => {
                    if (!x.needsGrad()) {
                        return;
                    }
                    const subGrad = Tensor.zeros(x.shape);
                    subGrad.data.set(grad.data.slice(chunkSize * i, chunkSize * (i + 1)));
                    x.backward(subGrad);
                });
            };
            return new Tensor(resultData, Shape.make(tensors.length, ...tensors[0].shape), backward);
        }

        static fromData(arr) {
            if (typeof arr === "number") {
                return new Tensor(new Float32Array([arr]), Shape.make(), null);
            }
            return Tensor.stackOuter(arr.map((x) => Tensor.fromData(x)));
        }

        static cat(tensors, axis) {
            // Sanity check for compatible shapes.
            const fillerShapes = tensors.map((x) => {
                return Shape.make(...x.shape.slice(0, axis), -1, ...x.shape.slice(axis + 1));
            });
            fillerShapes.forEach((x, i) => {
                if (x.length !== fillerShapes[0].length || !x.equals(fillerShapes[0])) {
                    throw new Error("incompatible shapes: " + tensors[0].shape + " and " +
                        tensors[i].shape);
                }
            });

            const newShape = Shape.make(
                ...tensors[0].shape.slice(0, axis),
                tensors.reduce((c, x) => c + x.shape[axis], 0),
                ...tensors[0].shape.slice(axis + 1),
            );
            const midSize = newShape[axis];
            const outerSize = newShape.slice(0, axis).numel();
            const innerSize = newShape.slice(axis + 1).numel();
            const result = Tensor.zeros(newShape);
            let offset = 0;
            tensors.forEach((x) => {
                const n = x.shape[axis];
                for (let i = 0; i < outerSize; ++i) {
                    for (let j = 0; j < n; ++j) {
                        for (let k = 0; k < innerSize; ++k) {
                            const c = x.data[(i * n + j) * innerSize + k];
                            result.data[(i * midSize + j + offset) * innerSize + k] = c;
                        }
                    }
                }
                offset += n;
            });
            result.backward = !tensors.some((x) => x.needsGrad()) ? null : (grad) => {
                let offset = 0;
                tensors.forEach((x) => {
                    const n = x.shape[axis];
                    if (x.needsGrad()) {
                        x.backward(grad.slice(axis, offset, offset + n));
                    }
                    offset += n;
                });
            };
            return result;
        }

        t() {
            if (this.shape.length !== 2) {
                throw new Error("can only transpose 2D array");
            }
            const result = Tensor.zeros(Shape.make(this.shape[1], this.shape[0]));
            for (let i = 0; i < this.shape[0]; ++i) {
                for (let j = 0; j < this.shape[1]; ++j) {
                    result.data[i + j * this.shape[0]] = this.data[i * this.shape[1] + j];
                }
            }
            result.backward = !this.needsGrad() ? null : (grad) => {
                this.backward(grad.t());
            };
            return result;
        }

        reshape(shape) {
            shape = shape.infer(this.shape.numel());
            if (shape.numel() !== this.shape.numel()) {
                throw new Error("old shape " + this.shape + " incompatible with " + shape);
            }
            const backward = !this.needsGrad() ? null : (grad) => {
                this.backward(grad.reshape(this.shape));
            };
            return new Tensor(this.data, shape, backward);
        }

        sum(axis) {
            if (axis < 0) {
                axis += this.shape.length;
            }
            if (axis >= this.shape.length || axis < 0) {
                throw new Error("axis " + axis + " out of range");
            }
            const n = this.shape[axis];
            let innerSize = 1;
            let outerSize = 1;
            this.shape.forEach((x, i) => {
                if (i < axis) {
                    outerSize *= x;
                } else if (i > axis) {
                    innerSize *= x;
                }
            });
            const data = new Float32Array(innerSize * outerSize);
            for (let i = 0; i < outerSize; ++i) {
                for (let j = 0; j < n; ++j) {
                    for (let k = 0; k < innerSize; ++k) {
                        data[k + i * innerSize] += this.data[(i * n + j) * innerSize + k];
                    }
                }
            }
            const newShape = Shape.make(
                ...this.shape.slice(0, axis),
                ...this.shape.slice(axis + 1, this.shape.length),
            );
            const backward = (!this.needsGrad() ? null : (grad) => {
                this.backward(grad.unsqueeze(axis).repeat(axis, n));
            });
            return new Tensor(data, newShape, backward);
        }

        unsqueeze(axis) {
            if (axis < 0) {
                axis += this.shape.length + 1;
            }
            const shape = Shape.make(...this.shape.slice(0, axis), 1, ...this.shape.slice(axis));
            return this.reshape(shape);
        }

        repeat(axis, reps) {
            const shape = Shape.make(
                ...this.shape.slice(0, axis),
                this.shape[axis] * reps,
                ...this.shape.slice(axis + 1),
            );
            const result = Tensor.zeros(shape);
            const outerSize = this.shape.slice(0, axis).numel();
            const innerSize = this.shape.slice(axis).numel();
            for (let i = 0; i < outerSize; ++i) {
                for (let j = 0; j < innerSize; ++j) {
                    const x = this.data[i * innerSize + j];
                    for (let k = 0; k < reps; ++k) {
                        result.data[(i * reps + k) * innerSize + j] = x;
                    }
                }
            }
            result.backward = !this.needsGrad() ? null : (grad) => {
                const extShape = Shape.make(
                    ...this.shape.slice(0, axis),
                    reps,
                    this.shape[axis],
                    ...this.shape.slice(axis + 1),
                )
                this.backward(grad.reshape(extShape).sum(axis));
            };
            return result;
        }

        slice(axis, start, end) {
            if (axis < 0) {
                axis += this.shape.length;
            }
            if (axis < 0 || axis >= this.shape.length) {
                throw new Error("axis out of bounds");
            }
            if (typeof end === "undefined") {
                end = this.shape[axis];
            }
            if (end < start || start < 0 || end > this.shape[axis]) {
                throw new Error('invalid range');
            }
            const outerSize = this.shape.slice(0, axis).numel();
            const innerSize = this.shape.slice(axis + 1).numel();
            const midSize = this.shape[axis];
            const result = Tensor.zeros(Shape.make(
                ...this.shape.slice(0, axis),
                end - start,
                ...this.shape.slice(axis + 1),
            ));
            for (let i = 0; i < outerSize; ++i) {
                for (let j = 0; j < end - start; ++j) {
                    for (let k = 0; k < innerSize; ++k) {
                        const c = this.data[(i * midSize + j + start) * innerSize + k];
                        result.data[(i * (end - start) + j) * innerSize + k] = c;
                    }
                }
            }
            if (this.needsGrad()) {
                throw new Error('slice does not support gradients');
            }
            return result;
        }

        add(other) {
            if (!this.shape.equals(other.shape)) {
                throw new Error("element-wise operation requires equal shape");
            }
            const res = this.detach().clone();
            other.data.forEach((x, i) => {
                res.data[i] += x;
            });
            res.backward = !(this.needsGrad() || other.needsGrad()) ? null : (grad) => {
                if (this.needsGrad()) {
                    this.backward(grad);
                }
                if (other.needsGrad()) {
                    other.backward(grad);
                }
            };
            return res;
        }

        mul(other) {
            if (!this.shape.equals(other.shape)) {
                throw new Error("element-wise operation requires equal shape");
            }
            const res = this.detach().clone();
            other.data.forEach((x, i) => {
                res.data[i] *= x;
            });
            res.backward = !(this.needsGrad() || other.needsGrad()) ? null : (grad) => {
                if (this.needsGrad()) {
                    this.backward(grad.mul(other.detach()));
                }
                if (other.needsGrad()) {
                    other.backward(grad.mul(this.detach()));
                }
            };
            return res;
        }

        needsGrad() {
            return this.backward !== null;
        }

        detach() {
            return new Tensor(this.data, this.shape, null);
        }

        clone() {
            const copyData = new Float32Array(this.data.length);
            copyData.set(this.data);
            return new Tensor(copyData, this.shape, this.backward);
        }
    }

    class Linear {
        constructor(weight, bias) {
            this.weight = weight;
            this.bias = bias;
            if (
                weight.shape.length !== 2 ||
                bias.shape.length !== 1 ||
                bias.shape[0] != weight.shape[1]
            ) {
                throw new Error("invalid shapes: " + weight.shape + ", " + bias.shape);
            }
            this.numOutput = weight.shape[0];
            this.numInput = weight.shape[1];
        }

        forward(x) {
            return addBias(matmul(x, this.weight), this.bias);
        }
    }

    function matmul(m1, m2) {
        if (m1.shape.length !== 2 || m2.shape.length !== 2 || m1.shape[1] !== m2.shape[0]) {
            throw new Error("invalid input shapes: " + m1.shape + ", " +
                m2.shape);
        }
        // ij,jk -> ik
        const sizeI = m1.shape[0];
        const sizeJ = m1.shape[1];
        const sizeK = m2.shape[1];
        const product = new Float32Array(sizeI * sizeK);
        for (let i = 0; i < sizeI; ++i) {
            for (let j = 0; j < sizeJ; ++j) {
                const w = m1.data[i * sizeJ + j];
                for (let k = 0; k < sizeK; ++k) {
                    const w1 = m2.data[j * sizeK + k];
                    product[i * sizeK + k] += w * w1;
                }
            }
        }
        const backward = (!m1.needsGrad() && !m2.needsGrad()) ? null : (grad) => {
            // ij,jk->ik
            // grad for m1 is ij = out_grad * m2.t()
            // grad for m2 is jk = m1.t() * out_grad
            if (m1.needsGrad()) {
                m1.backward(matmul(grad, m2.detach().t()));
            }
            if (m2.needsGrad()) {
                m2.backward(matmul(m1.detach().t(), grad));
            }
        };
        return new Tensor(product, Shape.make(m1.shape[0], m2.shape[1]), backward);
    }

    function addBias(x, y) {
        if (x.shape.length !== 2 || y.shape.length !== 1 || x.shape[1] !== y.shape[0]) {
            throw new Error("invalid shapes: " + x.shape + ", " + y.shape);
        }
        return x.add(y.unsqueeze(0).repeat(0, x.shape[0]));
    }

    window.nn = {
        Shape: Shape,
        Tensor: Tensor,
        Linear: Linear,
        matmul: matmul,
        addBias: addBias,
    };

})();
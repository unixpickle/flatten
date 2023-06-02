(function () {

    class Shape extends Array {
        toString() {
            return "Shape(" + this.join(", ") + ")"
        }

        numel() {
            return this.reduce((cur, v) => cur * v, 1);
        }

        infer(numel) {
            const numNeg1 = this.reduce((c, v) => c + (v === -1 ? 1 : 0));

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
            this.data = data;
            this.shape = shape;
            this.backward = backward || null;
        }

        static zeros(shape) {
            const data = new Float32Array(shape.numel());
            return new Tensor(data, shape, null);
        }

        t() {
            if (this.shape.length !== 2) {
                throw new Error("can only transpose 2D array");
            }
            const result = Tensor.zeros(new Shape(this.shape[1], this.shape[0]));
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
            const shape = shape.infer(this.shape.numel());
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
            const newShape = new Shape(
                ...this.shape.slice(0, i),
                ...this.shape.slice(i + 1, this.shape.length),
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
            const shape = new Shape(...this.shape.slice(0, axis), 1, ...this.shape.slice(axis));
            return this.reshape(shape);
        }

        repeat(axis, reps) {
            const shape = new Shape(
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
                const extShape = new Shape(
                    ...this.shape.slice(0, axis),
                    reps,
                    this.shape[axis],
                    ...this.shape.slice(axis + 1),
                )
                this.backward(grad.reshape(extShape).sum(axis));
            };
            return result;
        }

        add(other) {
            if (!this.shape.equals(other.shape)) {
                throw new Error("element-wise operation requires equal shape");
            }
            const res = this.detach().clone();
            res.data.forEach((x, i) => {
                return res.data[x] += other.data[i];
            });
            res.backward = !this.needsGrad() && other.needsGrad() ? null : (grad) => {
                this.backward(grad);
                other.backward(grad);
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
                bias.shape[0] != weight.shape[0]
            ) {
                throw new Error("invalid shapes: " + weight.shape + ", " + bias.shape);
            }
            this.n_output = weight.shape[0];
            this.n_input = weight.shape[1];
        }

        forward(x) {
            return addBias(matmul(this.weight, x), this.bias);
        }
    }

    function matmul(m1, m2) {
        if (m1.shape.length !== 2 || m2.shape.length !== 2 || m1.shape[1] !== m2.shape[0]) {
            throw new Error("invalid input shapes: " + m1.shape + ", " +
                m2.shape);
        }
        const bs = m1.shape[0];
        const n_input = m1.shape[1];
        const n_output = m2.shape[1];
        const product = new Float32Array(bs * n_output);
        for (let i = 0; i < n_output; ++i) {
            for (let j = 0; j < n_input; ++j) {
                const w = m1.data[i * this.n_input + j];
                for (let k = 0; k < bs; ++k) {
                    product[k * n_output + i] += w * m2.data[k * bs + j];
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
        return new Tensor(product, new Shape(m1.shape[0], m2.shape[1]), backward);
    }

    function addBias(x, y) {
        if (x.shape.length !== 2 || y.shape.length !== 1 || x.shape[1] !== y.shape[0]) {
            throw new Error("invalid shapes: " + x.shape + ", " + y.shape);
        }
        return x.add(y.unsqueeze(0).repeat(0, x.shape[0]));
    }

})();
(function () {

    class Tensor {
        constructor(data, shape, backward) {
            this.data = data;
            this.shape = shape;
            this.backward = backward || null;
        }

        t() {
            if (this.shape.length !== 2) {
                throw new Error("can only transpose 2D array");
            }
            const result = new Float32Array(this.data.length);
            for (let i = 0; i < this.shape[0]; ++i) {
                for (let j = 0; j < this.shape[1]; ++j) {
                    result[i + j * this.shape[0]] = this.data[i * this.shape[1] + j];
                }
            }
            const backward = !this.needsGrad() ? null : (grad) => {
                this.backward(grad.t());
            };
            return new Tensor(result, shape, backward);
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
            throw new Error("invalid input shapes: " + m1.shape.join(",") + ", " +
                m2.shape.join(","));
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
        return new Tensor(product, [m1.shape[0], m2.shape[1]], backward);
    }

    function addBias(x, y) {
        if (x.shape.length !== 2 || y.shape.length !== 1 || x.shape[1] !== y.shape[0]) {
            throw new Error("invalid shapes: " + x.shape.join(",") + ", " + y.shape.join(","));
        }
        const sum = x.detach().clone();
        for (let i = 0; i < x.shape[0]; ++i) {
            const offset = i * x.shape[1];
            for (let j = 0; j < x.shape[1]; ++j) {
                sum.data[offset + j] += y.data[j];
            }
        }
        sum.backward = (!x.needsGrad() && !y.needsGrad() ? null : (grad) => {
            if (x.needsGrad()) {
                x.backward(grad);
            }
            if (y.needsGrad()) {
                const g = new Float32Array(y.data.length);
                for (let i = 0; i < x.shape[0]; i++) {
                    for (let j = 0; j < x.shape[1]; j++) {
                        g.data[j] += grad.data[i * x.shape[1] + j];
                    }
                }
                y.backward(new Tensor(g, y.shape, null));
            }
        });
        return sum;
    }

})();
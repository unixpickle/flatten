class Shape extends Array<number> {
    static make(...args: number[]) {
        return Shape.from(args) as Shape;
    }

    toString(): string {
        return "Shape(" + this.join(", ") + ")"
    }

    numel(): number {
        let res = 1;
        for (let i = 0; i < this.length; ++i) {
            res *= this[i];
        }
        return res;
    }

    infer(numel: number) {
        const numNeg1 = this.reduce((c, v) => c + (v === -1 ? 1 : 0), 0);

        if (numNeg1 > 1) {
            throw new Error("too many unknown entries in shape: " + this);
        }

        if (numNeg1 === 1) {
            const product = this.reduce((c, v) => c * (v === -1 ? 1 : v), 1);
            if (numel % product !== 0) {
                throw new Error("incompatible shape " + this + " for " + numel + " elements");
            }
            const res = this.copy();
            res[this.indexOf(-1)] = numel / product;
            return res;
        }

        return this;
    }

    equals(other: Shape) {
        return this.length === other.length && !this.some((x, i) => x !== other[i]);
    }

    sizesAroundAxis(axis: number): [number, number, number] {
        const midSize = this[axis];
        let outerSize = 1;
        for (let i = 0; i < axis; ++i) {
            outerSize *= this[i];
        }
        let innerSize = 1;
        for (let i = axis + 1; i < this.length; ++i) {
            innerSize *= this[i];
        }
        return [outerSize, midSize, innerSize];
    }

    copy(): Shape {
        return Shape.from(this) as Shape;
    }

    slice(...args: number[]): Shape {
        return super.slice.apply(this, args) as Shape;
    }
}

type TensorNativeData = number | TensorNativeData[];

class Tensor {
    constructor(public data: Float32Array, public shape: Shape, public backward?: (grad: Tensor) => void) {
        if (data.length !== shape.numel()) {
            throw new Error("data size " + data.length + " does not match shape " + shape);
        }
        this.backward = backward || null;
    }

    static zeros(shape: Shape): Tensor {
        const data = new Float32Array(shape.numel());
        return new Tensor(data, shape, null);
    }

    static ones(shape: Shape): Tensor {
        const res = Tensor.zeros(shape);
        res.data.fill(1);
        return res;
    }

    static randn(shape: Shape): Tensor {
        const res = Tensor.zeros(shape);
        for (let i = 0; i < res.data.length; ++i) {
            // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
            const u = 1 - Math.random();
            const v = Math.random();
            const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            res.data[i] = z;
        }
        return res;
    }

    static stackOuter(tensors: Tensor[]): Tensor {
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
        const backward = !tensors.some((x) => x.needsGrad()) ? null : (grad: Tensor) => {
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

    static fromData(arr: TensorNativeData): Tensor {
        if (typeof arr === "number") {
            return new Tensor(new Float32Array([arr]), Shape.make(), null);
        }
        return Tensor.stackOuter(arr.map((x) => Tensor.fromData(x)));
    }

    static cat(tensors: Tensor[], axis: number) {
        if (axis < 0) {
            throw new Error("negative axis is not supported");
        }
        // Sanity check for compatible shapes.
        for (let i = 1; i < tensors.length; ++i) {
            const s0 = tensors[0].shape;
            const s1 = tensors[i].shape;
            if (s0.length !== s1.length) {
                throw new Error("incompatible shapes: " + s0 + " and " + s1);
            }
            for (let j = 0; j < s0.length; ++j) {
                if (j != axis) {
                    if (s0[j] !== s1[j]) {
                        throw new Error("incompatible shapes: " + s0 + " and " + s1 +
                            " differ in dimension " + j);
                    }
                }
            }
        }

        const newShape = tensors[0].shape.copy();
        for (let i = 1; i < tensors.length; ++i) {
            newShape[axis] += tensors[i].shape[axis];
        }

        const [outerSize, midSize, innerSize] = newShape.sizesAroundAxis(axis);
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

    toList(): TensorNativeData {
        if (this.shape.length === 0) {
            return this.data[0];
        } else if (this.shape.length === 1) {
            return Array.from(this.data);
        } else {
            const innerShape = this.shape.slice(1);
            const innerSize = innerShape.numel();
            const res = [];
            for (let i = 0; i < this.shape[0]; ++i) {
                const subData = this.data.slice(i * innerSize, (i + 1) * innerSize);
                const inner = new Tensor(subData, innerShape, null);
                res.push(inner.toList());
            }
            return res;
        }
    }

    t(): Tensor {
        if (this.shape.length !== 2 && this.shape.length !== 3) {
            throw new Error("can only transpose 2D or 3D array");
        }
        let result;
        if (this.shape.length === 2) {
            result = Tensor.zeros(Shape.make(this.shape[1], this.shape[0]));
            for (let i = 0; i < this.shape[0]; ++i) {
                for (let j = 0; j < this.shape[1]; ++j) {
                    result.data[i + j * this.shape[0]] = this.data[i * this.shape[1] + j];
                }
            }
        } else if (this.shape.length === 3) {
            result = Tensor.zeros(
                Shape.make(this.shape[0], this.shape[2], this.shape[1]),
            );
            for (let i = 0; i < this.shape[0]; ++i) {
                for (let j = 0; j < this.shape[1]; ++j) {
                    for (let k = 0; k < this.shape[2]; ++k) {
                        const src = (i * this.shape[1] + j) * this.shape[2] + k;
                        const dst = (i * result.shape[1] + k) * result.shape[2] + j;
                        result.data[dst] = this.data[src];
                    }
                }
            }
        }
        result.backward = !this.needsGrad() ? null : (grad) => {
            this.backward(grad.t());
        };
        return result;
    }

    reshape(shape: Shape): Tensor {
        shape = shape.infer(this.shape.numel());
        if (shape.numel() !== this.shape.numel()) {
            throw new Error("old shape " + this.shape + " incompatible with " + shape);
        }
        const backward = !this.needsGrad() ? null : (grad: Tensor) => {
            this.backward(grad.reshape(this.shape));
        };
        return new Tensor(this.data, shape, backward);
    }

    sum(axis?: number): Tensor {
        if (typeof axis === "undefined") {
            return this._sumAll();
        }
        if (axis < 0) {
            axis += this.shape.length;
        }
        if (axis >= this.shape.length || axis < 0) {
            throw new Error("axis " + axis + " out of range");
        }
        const n = this.shape[axis];
        const [outerSize, _midSize, innerSize] = this.shape.sizesAroundAxis(axis);
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
        const backward = (!this.needsGrad() ? null : (grad: Tensor) => {
            this.backward(grad.unsqueeze(axis).repeat(axis, n));
        });
        return new Tensor(data, newShape, backward);
    }

    _sumAll(): Tensor {
        const res = Tensor.zeros(new Shape());
        res.data[0] = this.data.reduce((total, x) => total + x, 0);
        res.backward = !this.needsGrad() ? null : (grad) => {
            const repeated = Tensor.zeros(this.shape);
            for (let i = 0; i < repeated.data.length; ++i) {
                repeated.data[i] = grad.data[0];
            }
            this.backward(repeated);
        };
        return res;
    }

    mean(axis?: number): Tensor {
        if (typeof axis === "undefined") {
            return this._sumAll().scale(1 / this.shape.numel());
        }
        if (axis < 0) {
            axis += this.shape.length;
        }
        if (axis >= this.shape.length || axis < 0) {
            throw new Error("axis " + axis + " out of range");
        }
        return this.sum(axis).scale(1 / this.shape[axis]);
    }

    unsqueeze(axis: number): Tensor {
        if (axis < 0) {
            axis += this.shape.length + 1;
        }
        const shape = new Shape(this.shape.length + 1);
        for (let i = 0; i < axis; ++i) {
            shape[i] = this.shape[i];
        }
        shape[axis] = 1;
        for (let i = axis; i < this.shape.length; i++) {
            shape[i + 1] = this.shape[i];
        }
        return this.reshape(shape);
    }

    repeat(axis: number, reps: number): Tensor {
        const shape = Shape.make(
            ...this.shape.slice(0, axis),
            this.shape[axis] * reps,
            ...this.shape.slice(axis + 1),
        );
        const result = Tensor.zeros(shape);
        const [outerSize, _midSize, innerSize] = this.shape.sizesAroundAxis(axis);
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

    slice(axis: number, start: number, end?: number): Tensor {
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
            throw new Error("invalid range");
        }
        const [outerSize, midSize, innerSize] = this.shape.sizesAroundAxis(axis);
        const newShape = this.shape.copy();
        newShape[axis] = end - start;
        const result = Tensor.zeros(newShape);
        for (let i = 0; i < outerSize; ++i) {
            for (let j = 0; j < end - start; ++j) {
                for (let k = 0; k < innerSize; ++k) {
                    const c = this.data[(i * midSize + j + start) * innerSize + k];
                    result.data[(i * (end - start) + j) * innerSize + k] = c;
                }
            }
        }
        result.backward = !this.needsGrad() ? null : (grad) => {
            const outGrad = Tensor.zeros(this.shape);
            for (let i = 0; i < outerSize; ++i) {
                for (let j = 0; j < end - start; ++j) {
                    for (let k = 0; k < innerSize; ++k) {
                        const c = grad.data[(i * (end - start) + j) * innerSize + k];
                        outGrad.data[(i * midSize + j + start) * innerSize + k] = c;
                    }
                }
            }
            this.backward(outGrad);
        };
        return result;
    }

    add(other: Tensor): Tensor {
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

    sub(other: Tensor): Tensor {
        return this.add(other.scale(-1));
    }

    mul(other: Tensor): Tensor {
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

    scale(s: number): Tensor {
        const res = this.detach().clone();
        for (let i = 0; i < res.data.length; ++i) {
            res.data[i] = s * res.data[i];
        }
        res.backward = !this.needsGrad() ? null : (grad) => {
            this.backward(grad.scale(s));
        };
        return res;
    }

    addScalar(s: number): Tensor {
        const res = this.clone();
        for (let i = 0; i < res.data.length; ++i) {
            res.data[i] = res.data[i] + s;
        }
        return res;
    }

    sin(): Tensor {
        const res = this.detach().clone();
        for (let i = 0; i < res.data.length; ++i) {
            res.data[i] = Math.sin(res.data[i]);
        }
        res.backward = !this.needsGrad() ? null : (grad) => {
            this.backward(grad.mul(this.detach().cos()));
        };
        return res;
    }

    cos(): Tensor {
        const res = this.detach().clone();
        for (let i = 0; i < res.data.length; ++i) {
            res.data[i] = Math.cos(res.data[i]);
        }
        res.backward = !this.needsGrad() ? null : (grad) => {
            this.backward(grad.mul(this.detach().sin()).scale(-1));
        };
        return res;
    }

    pow(p: number): Tensor {
        const res = this.detach().clone();
        for (let i = 0; i < res.data.length; ++i) {
            res.data[i] = Math.pow(res.data[i], p);
        }
        res.backward = !this.needsGrad() ? null : (grad) => {
            this.backward(grad.mul(this.pow(p - 1).scale(p)));
        };
        return res;
    }

    exp(): Tensor {
        const res = this.detach().clone();
        for (let i = 0; i < res.data.length; ++i) {
            res.data[i] = Math.exp(res.data[i]);
        }
        res.backward = !this.needsGrad() ? null : (grad) => {
            this.backward(grad.mul(res));
        };
        return res;
    }

    relu(): Tensor {
        const res = this.detach().clone();
        for (let i = 0; i < res.data.length; ++i) {
            if (res.data[i] < 0) {
                res.data[i] = 0;
            }
        }
        res.backward = !this.needsGrad() ? null : (grad) => {
            const outGrad = grad.clone();
            for (let i = 0; i < res.data.length; ++i) {
                if (res.data[i] === 0) {
                    outGrad.data[i] = 0;
                }
            }
            this.backward(outGrad);
        };
        return res;
    }

    needsGrad(): boolean {
        return this.backward !== null;
    }

    detach(): Tensor {
        return new Tensor(this.data, this.shape, null);
    }

    clone(): Tensor {
        const copyData = new Float32Array(this.data.length);
        copyData.set(this.data);
        return new Tensor(copyData, this.shape, this.backward);
    }

    accumGrad(f: (inputView: Tensor) => Tensor) {
        const input = this.detach();
        let totalGrad: Tensor = null;
        input.backward = !this.needsGrad() ? null : (x) => {
            if (!totalGrad) {
                totalGrad = x;
            } else {
                totalGrad = totalGrad.add(x);
            }
        };

        const output = f(input);
        const newOutput = output.detach();

        newOutput.backward = !this.needsGrad() ? null : (g) => {
            totalGrad = null;
            output.backward(g)
            this.backward(totalGrad);
        };

        return newOutput;
    }

    avgPool2d(size: number): Tensor {
        if (this.shape.length !== 4 || this.shape[2] % size || this.shape[3] % size) {
            throw new Error("invalid shape for avg pool of size " + size + ": " + this.shape);
        }
        if (this.needsGrad()) {
            throw new Error("average pooling does not currently support gradients");
        }
        const result = Tensor.zeros(
            Shape.make(
                this.shape[0],
                this.shape[1],
                this.shape[2] / size,
                this.shape[3] / size,
            ),
        );
        let inIndex = 0;
        const mult = 1 / (size * size);
        for (let i = 0; i < this.shape[0]; i++) {
            for (let j = 0; j < this.shape[1]; j++) {
                for (let k = 0; k < this.shape[2]; k++) {
                    for (let l = 0; l < this.shape[3]; l++) {
                        const outK = Math.floor(k / size);
                        const outL = Math.floor(l / size);
                        result.data[
                            ((i * result.shape[1] + j) * result.shape[2] + outK)
                            * result.shape[3] + outL
                        ] += this.data[inIndex++] * mult;
                    }
                }
            }
        }
        return result;
    }
}

abstract class TensorLayer {
    abstract forward(x: Tensor): Tensor;
}

class Linear extends TensorLayer {
    public numInput: number;
    public numOutput: number;

    constructor(public weight: Tensor, public bias: Tensor) {
        super();
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

    forward(x: Tensor): Tensor {
        return addBias(matmul(x, this.weight), this.bias);
    }
}

class Conv2d extends TensorLayer {
    public stride: number;

    constructor(public weight: Tensor, public bias: Tensor, stride?: number) {
        super();
        this.stride = stride || 1;
        if (
            weight.shape.length !== 4 ||
            bias.shape.length !== 1 ||
            bias.shape[0] != weight.shape[0]
        ) {
            throw new Error("invalid shapes: " + weight.shape + ", " + bias.shape);
        }
    }

    forward(x: Tensor): Tensor {
        return conv2d(this.weight, this.bias, x, this.stride);
    }
}

class AvgAndFlatten extends TensorLayer {
    forward(x: Tensor): Tensor {
        return x.reshape(Shape.make(x.shape[0], x.shape[1], -1)).mean(2);
    }
}

class ReLU extends TensorLayer {
    forward(x: Tensor): Tensor {
        return x.relu();
    }
}

class Sequential extends TensorLayer {
    constructor(public layers: TensorLayer[]) {
        super();
    }

    forward(x: Tensor): Tensor {
        let h = x;
        this.layers.forEach((l) => {
            h = l.forward(h);
        });
        return h;
    }
}

function matmul(m1: Tensor, m2: Tensor): Tensor {
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
    const backward = (!m1.needsGrad() && !m2.needsGrad()) ? null : (grad: Tensor) => {
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

function conv2d(weight: Tensor, bias: Tensor, images: Tensor, stride: number) {
    if (
        weight.shape.length !== 4 ||
        bias.shape.length !== 1 ||
        images.shape.length !== 4 ||
        weight.shape[1] !== images.shape[1] ||
        weight.shape[0] !== bias.shape[0]
    ) {
        throw new Error("invalid shapes: weight=" + weight + " bias=" + bias + " images=" + images);
    }
    const padded = zeroPadImage(images);
    const patches = imagePatches(padded, weight.shape[2], weight.shape[3], stride);

    // output shape [B * H * W x C]
    const wOut = matmul(
        patches.reshape(Shape.make(-1, patches.shape[3])), // [B*H*W x P]
        weight.reshape(Shape.make(weight.shape[0], -1)).t(), // [P x C]
    );
    const bOut = wOut.add(bias.unsqueeze(0).repeat(0, wOut.shape[0]));

    // Convert result into [B x C x H x W]
    return bOut
        .reshape(Shape.make(patches.shape[0], patches.shape[1] * patches.shape[2], -1))
        .t()
        .reshape(Shape.make(patches.shape[0], -1, patches.shape[1], patches.shape[2]));

}

function zeroPadImage(images: Tensor): Tensor {
    const shape = Shape.make(images.shape[0], images.shape[1], images.shape[2] + 2, images.shape[3] + 2);
    const results = Tensor.zeros(shape);
    for (let i = 0; i < images.shape[0]; i++) {
        for (let j = 0; j < images.shape[1]; j++) {
            for (let k = 0; k < images.shape[2]; k++) {
                for (let l = 0; l < images.shape[3]; l++) {
                    const src = (((i * images.shape[1]) + j) * images.shape[2] + k) *
                        images.shape[3] + l;
                    const dst = (((i * shape[1]) + j) * shape[2] + k + 1) * shape[3] + l + 1;
                    results.data[dst] = images.data[src];
                }
            }
        }
    }
    if (images.needsGrad()) {
        results.backward = (g) => {
            throw new Error("backward not supported for zero padding");
        }
    }
    return results;
}

function imagePatches(images: Tensor, patchH: number, patchW: number, stride: number): Tensor {
    const h = Math.floor((images.shape[2] - patchH) / stride + 1);
    const w = Math.floor((images.shape[3] - patchW) / stride + 1);
    const outShape = Shape.make(images.shape[0], h, w, images.shape[1] * patchH * patchW);
    const output = Tensor.zeros(outShape);
    for (let b = 0; b <= images.shape[0]; b++) {
        for (let i = 0; i <= h; i++) {
            for (let j = 0; j <= w; j++) {

                // Loop over inner patch.
                let outIdx = ((b * h + i) * w + j) * outShape[3];
                for (let c = 0; c < images.shape[1]; c++) {
                    for (let dy = 0; dy < patchH; dy++) {
                        const y = dy + i * stride;
                        for (let dx = 0; dx < patchW; dx++) {
                            const x = dx + j * stride;
                            const src = ((b * images.shape[1] + c) * images.shape[2] + y) *
                                images.shape[3] + x;
                            output.data[outIdx++] = images.data[src];
                        }
                    }
                }

            }
        }
    }
    if (images.needsGrad()) {
        output.backward = (_: Tensor) => {
            throw new Error("backward not supported for convolutional patching");
        }
    }
    return output;
}

function addBias(x: Tensor, y: Tensor): Tensor {
    if (x.shape.length !== 2 || y.shape.length !== 1 || x.shape[1] !== y.shape[0]) {
        throw new Error("invalid shapes: " + x.shape + ", " + y.shape);
    }
    return x.add(y.unsqueeze(0).repeat(0, x.shape[0]));
}

function rotation(axis: number, theta: Tensor): Tensor {
    if (!theta.shape.equals(new Shape())) {
        throw new Error("invalid theta shape (expected scalar): " + theta.shape);
    }
    const cos = Math.cos(theta.data[0]);
    const sin = Math.sin(theta.data[0]);
    const result = Tensor.zeros(Shape.make(3, 3));
    if (axis === 0) {
        result.data[0] = 1;
        result.data[4] = cos;
        result.data[5] = -sin;
        result.data[7] = sin;
        result.data[8] = cos;
        if (theta.needsGrad()) {
            result.backward = (grad) => {
                const downstream = Tensor.zeros(theta.shape);
                downstream.data[0] = (
                    grad.data[4] * -sin - grad.data[5] * cos + grad.data[7] * cos - grad.data[8] * sin
                );
                theta.backward(downstream);
            };
        }
    } else if (axis === 1) {
        result.data[0] = cos;
        result.data[2] = sin;
        result.data[4] = 1;
        result.data[6] = -sin;
        result.data[8] = cos;
        if (theta.needsGrad()) {
            result.backward = (grad) => {
                const downstream = Tensor.zeros(theta.shape);
                downstream.data[0] = (
                    grad.data[0] * -sin + grad.data[2] * cos - grad.data[6] * cos
                    - grad.data[8] * sin
                );
                theta.backward(downstream);
            };
        }
    } else if (axis === 2) {
        result.data[0] = cos;
        result.data[1] = -sin;
        result.data[3] = sin;
        result.data[4] = cos;
        result.data[8] = 1;
        if (theta.needsGrad()) {
            result.backward = (grad) => {
                const downstream = Tensor.zeros(theta.shape);
                downstream.data[0] = (
                    grad.data[0] * -sin - grad.data[1] * cos + grad.data[3] * cos
                    - grad.data[4] * sin
                );
                theta.backward(downstream);
            };
        }
    } else {
        throw new Error("invalid axis: " + axis);
    }
    return result;
}

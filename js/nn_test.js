(function () {

    const nn = self.nn;

    function testLinear() {
        const weights = nn.Tensor.fromData([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        let weightGrad = null;
        weights.backward = (x) => weightGrad = x;

        const biases = nn.Tensor.fromData([-1, -3, -2]);
        let biasGrad = null;
        biases.backward = (x) => biasGrad = x;

        const inputs = nn.Tensor.fromData([[2, 4]]);
        let inputGrad = null;
        inputs.backward = (x) => inputGrad = x;

        const layer = new nn.Linear(weights, biases);
        const output = layer.forward(inputs);

        console.assert(output.shape.equals(nn.Shape.make(1, 3)));
        console.assert(output.data[0] === 17);
        console.assert(output.data[1] === 21);
        console.assert(output.data[2] === 28);

        const loss = output.sum();
        console.assert(loss.data[0] === 66);

        loss.backward(nn.Tensor.fromData(1));

        console.assert(weightGrad.shape.equals(weights.shape));
        console.assert(biasGrad.shape.equals(biases.shape));
        console.assert(inputGrad.shape.equals(inputs.shape));

        const wGrad = new Float32Array([2, 2, 2, 4, 4, 4]);
        const bGrad = new Float32Array([1, 1, 1]);
        const inGrad = new Float32Array([6, 15]);
        weightGrad.data.forEach((x, i) => {
            console.assert(wGrad[i] === x);
        });
        biasGrad.data.forEach((x, i) => {
            console.assert(bGrad[i] === x);
        });
        inputGrad.data.forEach((x, i) => {
            console.assert(inGrad[i] === x);
        });
        console.log('[Done] Linear');
    }

    function testSlice() {
        const x = nn.Tensor.fromData([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [-1, -2, -3],
                [-4, -5, -6],
            ],
            [
                [3, 3, 5],
                [5, -10, -20],
            ],
        ]);
        let xGrad = null;
        x.backward = (g) => xGrad = g;
        console.assert(x.shape.equals(nn.Shape.make(4, 2, 3)));

        const chunk = x.slice(0, 1, 3);
        assertEqual(chunk, nn.Tensor.fromData([
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [-1, -2, -3],
                [-4, -5, -6],
            ],
        ]));

        const chunk1 = x.slice(2, 1, 3);
        chunk1.backward(chunk1.detach().scale(-1));
        assertEqual(xGrad, nn.Tensor.fromData([
            [
                [0, 2, 3],
                [0, 5, 6],
            ],
            [
                [0, 8, 9],
                [0, 11, 12],
            ],
            [
                [0, -2, -3],
                [0, -5, -6],
            ],
            [
                [0, 3, 5],
                [0, -10, -20],
            ],
        ]).scale(-1));

        console.log('[Done] slice');
    }

    function testAccumGrad() {
        const x = nn.Tensor.fromData([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [-1, -2, -3],
                [-4, -5, -6],
            ],
            [
                [3, 3, 5],
                [5, -10, -20],
            ],
        ]);
        let xGrad = null;
        x.backward = (g) => xGrad = g;

        const combined = x.accumGrad((x) => {
            const chunk1 = x.slice(2, 1, 3);
            const chunk2 = x.slice(2, 0, 1);
            return nn.Tensor.cat([chunk2, chunk1], 2);
        });
        combined.backward(x.scale(-3));
        assertEqual(xGrad, x.scale(-3));

        console.log('[Done] accumGrad');
    }

    function testCat() {
        const x = nn.Tensor.fromData([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
            [
                [-1, -2, -3],
                [-4, -5, -6],
            ],
            [
                [3, 3, 5],
                [5, -10, -20],
            ],
        ]);

        const t1 = nn.Tensor.cat([x.slice(0, 0, 2), x.slice(0, 2, 3), x.slice(0, 3, 4)], 0);
        assertEqual(x, t1);

        const t2 = nn.Tensor.cat([x.slice(1, 0, 1), x.slice(1, 1)], 1);
        assertEqual(x, t2);

        // Test zero-sized tensor.
        const t3 = nn.Tensor.cat([x.slice(1, 0, 2), x.slice(1, 2, 2)], 1);
        assertEqual(x, t3);

        const t4 = nn.Tensor.cat([x.slice(2, 0, 2), x.slice(2, 2, 3)], 2);
        assertEqual(x, t4);

        const x1 = nn.Tensor.fromData([[[1], [2]], [[4], [5]]]);
        const x2 = nn.Tensor.fromData([[[-2]], [[-1]]]);
        let x1Grad = null;
        let x2Grad = null;
        x1.backward = (x) => x1Grad = x;
        x2.backward = (x) => x2Grad = x;

        const joined = nn.Tensor.cat([x1, x2], 1);
        const grad = nn.Tensor.fromData([[[-3], [-4], [-8]], [[-5], [-6], [-9]]]);
        joined.backward(grad);

        assertEqual(x1Grad, nn.Tensor.fromData([[[-3], [-4]], [[-5], [-6]]]));
        assertEqual(x2Grad, nn.Tensor.fromData([[[-8]], [[-9]]]));

        console.log('[Done] cat');
    }

    function testSinCos() {
        const t = nn.Tensor.fromData([1, 2, 3, 4]);
        let tGrad = null;
        t.backward = (g) => tGrad = g;

        t.sin().sum().backward(nn.Tensor.fromData(1));
        tGrad.data.forEach((x, i) => {
            const y = Math.cos(t.data[i]);
            console.assert(Math.abs(x - y) < 1e-5, x, y);
        });

        t.cos().sum().backward(nn.Tensor.fromData(1));
        tGrad.data.forEach((x, i) => {
            const y = -Math.sin(t.data[i]);
            console.assert(Math.abs(x - y) < 1e-5, x, y);
        });

        console.log('[Done] sin/cos');
    }

    function testPow() {
        const x = nn.Tensor.fromData([1, 2, 3]);
        let xGrad = null;
        x.backward = (g) => xGrad = g;

        const out = x.pow(3);
        assertEqual(out, nn.Tensor.fromData([1, 2 * 2 * 2, 3 * 3 * 3]));

        out.sum().backward(nn.Tensor.fromData(1));
        assertEqual(xGrad, nn.Tensor.fromData([3, 3 * 2 * 2, 3 * 3 * 3]));

        console.log('[Done] pow');
    }

    function testExp() {
        const x = nn.Tensor.fromData([1, 2, 3, -1]);
        let xGrad = null;
        x.backward = (g) => xGrad = g;

        const out = x.exp();
        out.data.forEach((y, i) => {
            console.assert(Math.abs(y - Math.exp(x.data[i])) < 1e-5, y, x.data[i]);
        });

        out.sum().backward(nn.Tensor.fromData(-3));
        xGrad.data.forEach((g, i) => {
            console.assert(Math.abs(g - -3 * out.data[i]) < 1e-5, g, x.data[i]);
        });

        console.log('[Done] exp')
    }

    function testReLU() {
        const x = nn.Tensor.fromData([1, 2, -4, 0, 3, -1]);
        let xGrad = null;
        x.backward = (g) => xGrad = g;

        const out = x.relu();
        assertEqual(out, nn.Tensor.fromData([1, 2, 0, 0, 3, 0]));

        out.sum().backward(nn.Tensor.fromData(-3));
        assertEqual(xGrad, nn.Tensor.fromData([-3, -3, 0, 0, -3, 0]));

        console.log('[Done] ReLU');
    }

    function testRotation() {
        for (let axis = 0; axis < 3; ++axis) {
            [-0.3, 0.3, Math.PI, Math.PI * 2].forEach((theta) => {
                const tensor = nn.rotation(axis, nn.Tensor.fromData(theta));

                // Make sure it's orthonormal.
                const product = nn.matmul(tensor, tensor.t());
                const identity = nn.Tensor.fromData([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
                identity.data.forEach((x, i) => {
                    console.assert(Math.abs(x - product.data[i]) < 1e-5, product)
                });

                // Make sure negative angle is the inverse.
                const invRot = nn.rotation(axis, nn.Tensor.fromData(-theta));
                invRot.t().data.forEach((x, i) => {
                    console.assert(Math.abs(x - tensor.data[i]) < 1e-5, tensor, invRot);
                });

                // Test backward pass with finite differences.
                const thetaT = nn.Tensor.fromData(theta);
                let thetaGrad = null;
                thetaT.backward = (g) => thetaGrad = g;

                const outGrad = nn.Tensor.fromData([
                    [1.0, -2.3, 3.1], [-0.53, 0.35, 0.837], [-0.7, 0.82, -0.9],
                ]);
                const objective = (x) => {
                    const mat = nn.rotation(axis, x);
                    return mat.mul(outGrad).sum();
                };

                objective(thetaT).backward(nn.Tensor.fromData(1));

                const epsilon = 1e-2;
                const o1 = objective(nn.Tensor.fromData(theta + epsilon)).data[0];
                const o2 = objective(nn.Tensor.fromData(theta - epsilon)).data[0];
                const approxGrad = (o1 - o2) / (2 * epsilon);

                console.assert(
                    // Even 1e-4 should work, but we don't want spurious failures.
                    Math.abs(approxGrad - thetaGrad.data[0]) < 1e-2,
                    approxGrad,
                    thetaGrad.data[0],
                );
            });
        }
        console.log('[Done] rotation');
    }

    function assertEqual(t1, t2) {
        console.assert(t1.shape.equals(t2.shape), t1.shape, t2.shape);
        const bad = t1.data.some((x, i) => x != t2.data[i]);
        console.assert(!bad, t1.data, t2.data);
    }

    self.nn.runTests = () => {
        testLinear();
        testSlice();
        testAccumGrad();
        testCat();
        testSinCos();
        testPow();
        testExp();
        testReLU();
        testRotation();
    };

})();
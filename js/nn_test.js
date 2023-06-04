(function () {

    const nn = window.nn;

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

        const loss = output.sum(0).sum(0);
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

    function testSplit() {
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
        console.log('[Done] split');
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

        console.log('[Done] cat');
    }

    function assertEqual(t1, t2) {
        console.assert(t1.shape.equals(t2.shape), t1.shape, t2.shape);
        const bad = t1.data.some((x, i) => x != t2.data[i]);
        console.assert(!bad, t1.data, t2.data);
    }

    window.nn.runTests = () => {
        testLinear();
        testSplit();
        testCat();
    };

})();
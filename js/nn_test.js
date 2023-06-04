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

    window.nn.runTests = () => {
        testLinear();
    };

})();
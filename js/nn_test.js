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
        const loss = output.sum(0).sum(0);
        loss.backward(nn.Tensor.fromData(1));
    }

    window.nn.runTests = () => {
        testLinear();
    };

})();
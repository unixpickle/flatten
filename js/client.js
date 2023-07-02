class ModelClient {
    constructor() {
        this._worker = new Worker("js/worker.js");
        this._callbacks = {};
        this._curRequestId = 0;
        this._worker.onmessage = (e) => {
            const msg = e.data;
            const [resolve, reject] = this._callbacks[msg.id];
            delete this._callbacks[msg.id];
            if (msg["error"]) {
                reject(new Error(msg.error));
            } else {
                resolve(msg.data);
            }
        };
    }

    async _call(method, args) {
        const reqId = this._curRequestId++;
        const promise = new Promise((resolve, reject) => {
            this._callbacks[reqId] = [resolve, reject];
        });
        this._worker.postMessage({
            id: reqId,
            method: method,
            args: args,
        });
        return promise;
    }

    async solve(points) {
        return this._call('solve', [points.map((p) => [p.x, p.y])]).then((sol) => {
            const solVec = nn.Tensor.fromData(sol);
            return nn.PerspectiveSolution.fromFlatVec(solVec);
        });
    }

    async predictStretch(imageData) {
        return this._call('predictStretch', [imageData]).then((preds) => preds[0]);
    }
}

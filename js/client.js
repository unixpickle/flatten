class ModelClient {
    constructor() {
        this._worker = new Worker("js/worker.js");
        this._callbacks = {};
        this._curRequestId = 0;
        this._worker.onmessage = (e) => {
            const msg = e.data;
            const [resolve, reject, statusFn] = this._callbacks[msg.id];
            if (msg["error"]) {
                reject(new Error(msg.error));
                delete this._callbacks[msg.id];
            } else if (msg["status"]) {
                statusFn(msg["status"]);
            } else {
                resolve(msg.data);
                delete this._callbacks[msg.id];
            }
        };
    }

    async _call(statusFn, method, args) {
        const reqId = this._curRequestId++;
        const promise = new Promise((resolve, reject) => {
            this._callbacks[reqId] = [resolve, reject, statusFn];
        });
        this._worker.postMessage({
            id: reqId,
            method: method,
            args: args,
        });
        return promise;
    }

    async solve(points, statusFn) {
        return this._call(statusFn, 'solve', [points.map((p) => [p.x, p.y])]).then((sol) => {
            const solVec = nn.Tensor.fromData(sol);
            return nn.PerspectiveSolution.fromFlatVec(solVec);
        });
    }

    async predictStretch(imageData, statusFn) {
        return this._call(statusFn, 'predictStretch', [imageData]).then((preds) => preds[0]);
    }
}

var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
class ModelClient {
    constructor() {
        this.worker = new Worker("js/worker.js");
        this.callbacks = {};
        this.curRequestId = 0;
        this.worker.onmessage = (e) => {
            const msg = e.data;
            const cb = this.callbacks[msg.id];
            if (msg["error"]) {
                cb.reject(new Error(msg.error));
                delete this.callbacks[msg.id];
            }
            else if (msg["status"]) {
                cb.status(msg["status"]);
            }
            else {
                cb.resolve(msg.data);
                delete this.callbacks[msg.id];
            }
        };
    }
    call(statusFn, method, args) {
        return __awaiter(this, void 0, void 0, function* () {
            const reqId = this.curRequestId++;
            const promise = new Promise((resolve, reject) => {
                this.callbacks[reqId] = { resolve: resolve, reject: reject, status: statusFn };
            });
            this.worker.postMessage({
                id: reqId,
                method: method,
                args: args,
            });
            return promise;
        });
    }
    solve(points, statusFn) {
        return __awaiter(this, void 0, void 0, function* () {
            const sol = yield this.call(statusFn, 'solve', [points.map((p) => [p.x, p.y])]);
            const solVec = Tensor.fromData(sol);
            return PerspectiveSolution.fromFlatVec(solVec);
        });
    }
    predictStretch(imageData, statusFn) {
        return __awaiter(this, void 0, void 0, function* () {
            const preds = yield this.call(statusFn, 'predictStretch', [imageData]);
            return preds[0];
        });
    }
    exportImage(solution, pixelSource, aspectRatio, sideLength) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.call(noopStatusFunc, 'exportImage', [
                solution.toFlatVec().toList(),
                pixelSource.serialize(),
                aspectRatio,
                sideLength,
            ]);
        });
    }
}
function noopStatusFunc(_) {
}
//# sourceMappingURL=client.js.map
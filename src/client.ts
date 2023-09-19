type StatusFunc = (_: string) => void;

interface ModelClientCallback {
    resolve: (_: any) => void;
    reject: (_: any) => void;
    status: StatusFunc;
}

class ModelClient {
    private worker: Worker;
    private callbacks: { [key: string]: ModelClientCallback };
    private curRequestId: number;

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
            } else if (msg["status"]) {
                cb.status(msg["status"]);
            } else {
                cb.resolve(msg.data);
                delete this.callbacks[msg.id];
            }
        };
    }

    private async call(statusFn: StatusFunc, method: string, args: any[]): Promise<any> {
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
    }

    async solve(points: Point2[], statusFn: StatusFunc): Promise<PerspectiveSolution> {
        const sol: number[] = await this.call(statusFn, 'solve', [points.map((p) => [p.x, p.y])]);
        const solVec = Tensor.fromData(sol);
        return PerspectiveSolution.fromFlatVec(solVec);
    }

    async predictStretch(imageData: TensorNativeData, statusFn: StatusFunc): Promise<number> {
        const preds: number[] = await this.call(statusFn, 'predictStretch', [imageData]);
        return preds[0];
    }

    async exportImage(solution: PerspectiveSolution, pixelSource: PixelSource, aspectRatio: number, sideLength: number) {
        return this.call(noopStatusFunc, 'exportImage', [
            solution.toFlatVec().toList(),
            pixelSource.serialize(),
            aspectRatio,
            sideLength,
        ]);
    }
}

function noopStatusFunc(_: string) {
}

interface SerializedPixelSource {
    imageData: Uint8ClampedArray;
    width: number;
    height: number;
}

class PixelSource {
    constructor(public imageData: Uint8ClampedArray, public width: number, public height: number) {
    }

    static fromCanvas(canvas: HTMLCanvasElement) {
        const ctx = canvas.getContext("2d");
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        return new PixelSource(data.data, canvas.width, canvas.height);
    }

    static deserialize(obj: SerializedPixelSource) {
        return new PixelSource(obj.imageData, obj.width, obj.height);
    }

    serialize(): SerializedPixelSource {
        return { imageData: this.imageData, width: this.width, height: this.height };
    }

    extractProjectedImage(solution: PerspectiveSolution, dstCanvas: HTMLCanvasElement | OffscreenCanvas) {
        const projector = solution.projector();

        const dst = dstCanvas.getContext("2d");
        const imgData = dst.createImageData(dstCanvas.width, dstCanvas.height);

        const [w, h] = solution.size.toList() as number[];
        const scaleX = w / dstCanvas.width;
        const scaleY = h / dstCanvas.height;

        const dstPoints = Tensor.zeros(Shape.make(dstCanvas.width, 2));
        for (let y = 0; y < dstCanvas.height; y++) {
            const scaledY = y * scaleY;
            for (let x = 0; x < dstPoints.shape[0]; x++) {
                dstPoints.data[x * 2] = x * scaleX;
                dstPoints.data[x * 2 + 1] = scaledY;
            }
            const srcPoints = projector(dstPoints);
            const sourcePixels = this.getPixels(srcPoints);
            const offset = y * sourcePixels.shape.numel();
            for (let i = 0; i < sourcePixels.data.length; i++) {
                imgData.data[offset + i] = sourcePixels.data[i];
            };
        }

        dst.putImageData(imgData, 0, 0);
    }

    getPixels(coords: Tensor): Tensor {
        const data = this.imageData;
        const eps = 1e-5; // make sure to never go out of bounds
        const xScale = (this.width - 1 - eps);
        const yScale = (this.height - 1 - eps);

        const output = Tensor.zeros(Shape.make(coords.shape[0], 4));
        for (let outIdx = 0; outIdx < coords.shape[0]; outIdx++) {
            const outOffset = outIdx * 4;
            const relX = coords.data[outIdx * 2];
            const relY = coords.data[outIdx * 2 + 1];

            const x = Math.max(0, Math.min(1, relX)) * xScale;
            const y = Math.max(0, Math.min(1, relY)) * yScale;
            const minX = Math.floor(x);
            const minY = Math.floor(y);

            const fracX = x - minX;
            const fracY = y - minY;

            for (let i = 0; i < 2; i++) {
                const wy = (i === 0 ? 1 - fracY : fracY);
                for (let j = 0; j < 2; j++) {
                    const wx = (j === 0 ? 1 - fracX : fracX);
                    const w = wx * wy;
                    const offset = ((minX + j) + (minY + i) * this.width) * 4;
                    for (let k = 0; k < 4; k++) {
                        output.data[outOffset + k] += w * data[offset + k];
                    }
                }
            }
        }
        for (let i = 0; i < output.data.length; i++) {
            output.data[i] = Math.round(output.data[i]);
        }
        return output;
    }
}
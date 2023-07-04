const HONING_AREA_SIZE = 0.05;

class App {
    constructor() {
        this.modelClient = new ModelClient();

        // State set once an image is picked
        this._img = null;
        this._scale = null;
        this._offsetX = 0;
        this._offsetY = 0;

        // State updated every time a point is picked.
        this._points = [];

        // Set when picking a point before honing.
        this._hoveringHoningPoint = null;

        // Set when honing a point after clicking a rough area.
        this._honingPoint = null;

        // Set to true when loading a solution.
        this._isLoading = false;

        this.canvas = document.getElementById("picker");
        this.canvas.addEventListener("dragover", (e) => this.handleDragOver(e), false);
        this.canvas.addEventListener("dragleave", (e) => this.canvas.classList.remove("dragging"));
        this.canvas.addEventListener("drop", (e) => this.handleDrop(e), false);
        this.canvas.addEventListener("mousedown", (e) => this.handleMouseDown(e));
        this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));
    }

    handleDragOver(e) {
        e.stopPropagation();
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        this.canvas.classList.add("dragging-file");
    }

    handleDrop(e) {
        this.canvas.classList.remove("dragging")
        e.stopPropagation();
        e.preventDefault();
        const files = e.dataTransfer.files;
        this.handleFile(files[0]);
        this.canvas.classList.add("dragging");
    }

    handleFile(f) {
        if (this._isLoading || !f.type.startsWith("image/")) {
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            var img = new Image();
            img.onload = () => this.resetImage(img);
            img.src = e.target.result;
        };
        reader.readAsDataURL(f);
    }

    promptFileUpload() {
        const input = document.createElement("input");
        input.type = "file";
        input.onchange = (e) => this.handleFile(e.target.files[0]);
        input.click();
    }

    _mouseEventPoint(e) {
        const x = e.offsetX;
        const y = e.offsetY;
        return new Point2(
            x / this.canvas.width,
            y / this.canvas.height,
        );
    }

    handleMouseDown(e) {
        this.handleClick(this._mouseEventPoint(e));
    }

    handleMouseMove(e) {
        this.handleHover(this._mouseEventPoint(e));
    }

    handleClick(point) {
        if (this._isLoading) {
            return;
        }

        if (!this._img) {
            this.promptFileUpload();
        } else if (this._honingPoint) {
            this._points.push(new Point2(
                this._honingPoint.x + HONING_AREA_SIZE * point.x,
                this._honingPoint.y + HONING_AREA_SIZE * point.y,
            ));
            this._honingPoint = null;
            if (this._points.length === 4) {
                this.solve();
            }
            this.draw();
        } else {
            this._hoveringHoningPoint = null;
            this._honingPoint = this.honingPointForCursor(point);
            this.draw();
        }
    }

    handleHover(point) {
        if (this._isLoading) {
            return;
        }

        if (this._img && !this._honingPoint) {
            this._hoveringHoningPoint = this.honingPointForCursor(point);
            this.draw();
        }
    }

    honingPointForCursor(point) {
        // TODO: clip to image bounds.
        return new Point2(point.x - HONING_AREA_SIZE / 2, point.y - HONING_AREA_SIZE / 2);
    }

    resetImage(img) {
        const iw = img.width;
        const ih = img.height;
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        this._scale = Math.min(cw / iw, ch / ih);
        this._offsetX = Math.round(cw - (this._scale * iw));
        this._offsetY = Math.round(ch - (this._scale * ih));
        this._img = img;
        this._honingPoint = null;
        this._hoveringHoningPoint = null;
        this._points = [];
        this.draw();
    }

    draw() {
        const ctx = this.canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (!this._img) {
            // TODO: draw prompt to drop files.
        } else if (!this._honingPoint) {
            ctx.drawImage(
                this._img,
                this._offsetX,
                this._offsetY,
                this._scale * this._img.width,
                this._scale * this._img.height,
            );
            if (this._hoveringHoningPoint) {
                ctx.strokeStyle = "green";
                ctx.beginPath();
                ctx.rect(
                    this._hoveringHoningPoint.x * this.canvas.width,
                    this._hoveringHoningPoint.y * this.canvas.width,
                    HONING_AREA_SIZE * this.canvas.width,
                    HONING_AREA_SIZE * this.canvas.height,
                );
                ctx.stroke();
            }
            this._points.forEach((p) => {
                const x = this.canvas.width * p.x;
                const y = this.canvas.height * p.y;
                ctx.fillStyle = "green";
                ctx.beginPath();
                ctx.arc(x, y, 5.0, 0, Math.PI * 2, false);
                ctx.fill();
            });
        } else {
            ctx.save();
            ctx.scale(1 / HONING_AREA_SIZE, 1 / HONING_AREA_SIZE);
            ctx.translate(this._offsetX, this._offsetY);
            ctx.translate(
                -this._honingPoint.x * this.canvas.width,
                -this._honingPoint.y * this.canvas.height,
            );
            ctx.drawImage(
                this._img,
                0,
                0,
                this._scale * this._img.width,
                this._scale * this._img.height,
            );
            ctx.restore();
        }
    }

    solve() {
        this._isLoading = true;
        let solution = null;
        this.modelClient.solve(this._points).then((s) => {
            solution = s;
            const dstCanvas = document.createElement("canvas");
            dstCanvas.width = 64;
            dstCanvas.height = 64;
            extractProjectedImage(solution, this.pixelSource(), dstCanvas);
            const imageData = canvasToTensor(dstCanvas).toList();
            return this.modelClient.predictStretch(imageData);
        }).then((stretch) => {
            const w = 1;
            const h = stretch;
            const scale = Math.min(200 / w, 200 / h);

            const dstCanvas = document.createElement("canvas");
            dstCanvas.width = Math.ceil(w * scale);
            dstCanvas.height = Math.ceil(h * scale);
            extractProjectedImage(solution, this.pixelSource(), dstCanvas);
            document.body.appendChild(dstCanvas);
        });
    }

    pixelSource() {
        const scale = Math.max(
            this._img.width / this.canvas.width,
            this._img.height / this.canvas.height,
        );
        const extractionCanvas = document.createElement("canvas");
        extractionCanvas.width = this.canvas.width * scale;
        extractionCanvas.height = this.canvas.height * scale;
        const ctx = extractionCanvas.getContext("2d");
        ctx.drawImage(
            this._img,
            this._offsetX * scale,
            this._offsetY * scale,
            this._scale * this._img.width * scale,
            this._scale * this._img.height * scale,
        );

        const data = ctx.getImageData(0, 0, extractionCanvas.width, extractionCanvas.height);
        return (relX, relY) => {
            const eps = 1e-5; // make sure to never go out of bounds
            const x = Math.max(0, Math.min(1, relX)) * (extractionCanvas.width - 1 - eps);
            const y = Math.max(0, Math.min(1, relY)) * (extractionCanvas.height - 1 - eps);
            const minX = Math.floor(x);
            const minY = Math.floor(y);

            const fracX = x - minX;
            const fracY = y - minY;

            const result = [0, 0, 0, 0];
            for (let i = 0; i < 2; i++) {
                const wy = (i === 0 ? 1 - fracY : fracY);
                for (let j = 0; j < 2; j++) {
                    const wx = (j === 0 ? 1 - fracX : fracX);
                    const w = wx * wy;
                    const offset = ((minX + j) + (minY + i) * extractionCanvas.width) * 4;
                    for (let k = 0; k < 4; k++) {
                        result[k] += w * data.data[offset + k];
                    }
                }
            }

            return result.map((x) => Math.round(x));
        };
    }
}

function extractProjectedImage(solution, src, dstCanvas) {
    const projector = solution.projector();

    const dst = dstCanvas.getContext("2d");
    const imgData = dst.createImageData(dstCanvas.width, dstCanvas.height);

    const [w, h] = solution.size.toList();
    const scaleX = w / dstCanvas.width;
    const scaleY = h / dstCanvas.height;

    for (let y = 0; y < dstCanvas.height; y++) {
        for (let x = 0; x < dstCanvas.width; x++) {
            const dstPoint = new Point2(x * scaleX, y * scaleY);
            const sourcePoint = projector(dstPoint);
            const sourcePixel = src(sourcePoint.x, sourcePoint.y);
            const idx = (x + y * dstCanvas.width) * 4;
            for (let i = 0; i < 4; i++) {
                imgData.data[idx + i] = sourcePixel[i];
            }
        }
    }

    dst.putImageData(imgData, 0, 0);
}

function canvasToTensor(canvas) {
    const imgData = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
    const result = nn.Tensor.zeros(nn.Shape.make(1, 3, canvas.height, canvas.width));
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            for (let k = 0; k < 3; k++) {
                const data = imgData.data[(y * canvas.width + x) * 4 + k] / 255;
                result.data[(k * canvas.height + y) * canvas.width + x] = data;
            }
        }
    }
    console.log(result);
    return result;
}

class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

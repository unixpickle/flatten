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

        this.draw();
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
        this._isLoading = false;
        this.draw();
    }

    draw() {
        const ctx = this.canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this._isLoading) {
            this.drawText("Loading...");
        } else if (!this._img) {
            this.drawText("Drop files or click here");
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

    drawText(text) {
        const ctx = this.canvas.getContext("2d");
        ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
        ctx.beginPath();
        ctx.rect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fill();

        ctx.fillStyle = "#777";
        ctx.font = "30px sans-serif";

        ctx.beginPath();
        ctx.textAlign = "center";
        ctx.fillText(text, this.canvas.width / 2, this.canvas.height / 2);
    }

    solve() {
        this._isLoading = true;
        const pixelSource = this.pixelSource();

        let solution = null;
        this.modelClient.solve(this._points).then((s) => {
            solution = s;
            const dstCanvas = document.createElement("canvas");
            dstCanvas.width = 64;
            dstCanvas.height = 64;
            extractProjectedImage(solution, pixelSource, dstCanvas);
            const imageData = canvasToTensor(dstCanvas).toList();
            return this.modelClient.predictStretch(imageData);
        }).then((stretch) => {
            const finishDialog = new FinishDialog(
                solution,
                pixelSource,
                stretch,
                Math.max(this._img.width, this._img.height),
            );
            this.canvas.style.display = "none";
            finishDialog.onClose = () => {
                finishDialog.hide();
                this.canvas.style.display = "block";
                this.resetImage(this._img);
            };
            finishDialog.show();
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

class FinishDialog {
    constructor(solution, pixelSource, aspectRatio, defaultSize) {
        this.solution = solution;
        this.pixelSource = pixelSource;

        this.element = document.getElementById("finish-dialog");
        this.previewContainer = this.element.getElementsByClassName("scaling-preview")[0];
        this.previewContainer.innerHTML = "";
        this.sideLength = this.element.getElementsByClassName("side-length")[0];
        this.aspectRatio = this.element.getElementsByClassName("aspect-ratio")[0];
        this.downloadButton = this.element.getElementsByClassName("download-button")[0];
        this.closeButton = this.element.getElementsByClassName("close-button")[0];

        this.onClose = () => null;
        this.closeButton.addEventListener("click", () => this.onClose());
        this.downloadButton.addEventListener("click", () => {
            this.downloadButton.href = this.downloadURL();
        }, false);

        this.aspectRatio.value = Math.log(aspectRatio);
        this.sideLength.value = defaultSize;

        this.previewCanvas = this.createCanvas(1, 256);
        this.previewCanvas.style.position = "absolute";
        this.updateAspectRatio();
        this.previewContainer.appendChild(this.previewCanvas);

        this.aspectRatio.addEventListener("input", () => this.updateAspectRatio());
    }

    createCanvas(aspectRatio, sideLength) {
        aspectRatio = aspectRatio || Math.exp(parseFloat(this.aspectRatio.value));
        sideLength = sideLength || parseInt(this.sideLength.value);

        const w = 1;
        const h = aspectRatio;
        const scale = Math.min(sideLength / w, sideLength / h);

        const dstCanvas = document.createElement("canvas");
        dstCanvas.width = w * scale;
        dstCanvas.height = h * scale;
        extractProjectedImage(this.solution, this.pixelSource, dstCanvas);
        return dstCanvas;
    }

    updateAspectRatio() {
        const ratio = Math.exp(parseFloat(this.aspectRatio.value));
        if (ratio > 1) {
            this.previewCanvas.style.width = (100 / ratio).toFixed(2) + "%";
            this.previewCanvas.style.left = (50 - 50 / ratio).toFixed(2) + "%";
            this.previewCanvas.style.height = "100%";
            this.previewCanvas.style.top = "0";
        } else {
            this.previewCanvas.style.height = (100 * ratio).toFixed(2) + "%";
            this.previewCanvas.style.top = (50 - 50 * ratio).toFixed(2) + "%";
            this.previewCanvas.style.width = "100%";
            this.previewCanvas.style.left = "0";
        }
    }

    downloadURL() {
        const canvas = this.createCanvas();
        // https://stackoverflow.com/questions/12796513/html5-canvas-to-png-file
        let dt = canvas.toDataURL("image/png");
        dt = dt.replace(/^data:image\/[^;]*/, "data:application/octet-stream");
        dt = dt.replace(
            /^data:application\/octet-stream/,
            "data:application/octet-stream;headers=Content-Disposition%3A%20attachment%3B%20filename=flattened.png",
        );
        return dt;
    }

    show() {
        this.element.style.display = "block";
    }

    hide() {
        this.element.style.display = "none";
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

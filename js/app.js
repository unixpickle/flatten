var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const HONING_AREA_SIZE = 0.05;
class App {
    constructor() {
        this.modelClient = new ModelClient();
        this.picker = new Picker();
        this.picker.onPick = (points) => this.solve(points);
    }
    solve(points) {
        return __awaiter(this, void 0, void 0, function* () {
            const statusFn = (x) => this.picker.showStatus(x);
            const pixelSource = this.picker.pixelSource();
            const solution = yield this.modelClient.solve(points, statusFn);
            const dstCanvas = document.createElement("canvas");
            dstCanvas.width = 128;
            dstCanvas.height = 128;
            pixelSource.extractProjectedImage(solution, dstCanvas);
            const imageData = canvasToTensor(dstCanvas).avgPool2d(2).toList();
            const stretch = yield this.modelClient.predictStretch(imageData, statusFn);
            const finishDialog = new FinishDialog(solution, pixelSource, stretch, this.picker.maxDimension(), this.modelClient);
            this.picker.hide();
            finishDialog.show();
            finishDialog.onClose = () => {
                finishDialog.hide();
                this.picker.show();
            };
        });
    }
}
class Picker {
    constructor() {
        this.img = null;
        this.scale = null;
        this.offsetX = 0;
        this.offsetY = 0;
        this.points = [];
        this.hoveringHoningPoint = null;
        this.honingPoint = null;
        this.isLoading = false;
        this.loadingStatus = null;
        this.container = document.getElementById("picker-container");
        this.canvas = document.getElementById("picker");
        this.canvas.addEventListener("dragover", (e) => this.handleDragOver(e), false);
        this.canvas.addEventListener("dragleave", (e) => this.canvas.classList.remove("dragging"));
        this.canvas.addEventListener("drop", (e) => this.handleDrop(e), false);
        this.canvas.addEventListener("click", (e) => {
            if (e.detail > 1) {
                return;
            }
            this.handleMouseDown(e);
        });
        this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));
        this.undoButton = this.container.getElementsByClassName("undo-button")[0];
        this.undoButton.addEventListener("click", () => this.undo());
        this.draw();
    }
    handleDragOver(e) {
        e.stopPropagation();
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        this.canvas.classList.add("dragging-file");
    }
    handleDrop(e) {
        this.canvas.classList.remove("dragging");
        e.stopPropagation();
        e.preventDefault();
        const files = e.dataTransfer.files;
        this.handleFile(files[0]);
        this.canvas.classList.add("dragging");
    }
    handleFile(f) {
        if (this.isLoading || !f.type.startsWith("image/")) {
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
        input.onchange = (e) => this.handleFile(input.files[0]);
        input.click();
    }
    _mouseEventPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.offsetX / (rect.width / this.canvas.width);
        const y = e.offsetY / (rect.height / this.canvas.height);
        return new Point2(x / this.canvas.width, y / this.canvas.height);
    }
    handleMouseDown(e) {
        this.handleClick(this._mouseEventPoint(e));
    }
    handleMouseMove(e) {
        this.handleHover(this._mouseEventPoint(e));
    }
    handleClick(point) {
        if (this.isLoading) {
            return;
        }
        if (!this.img) {
            this.promptFileUpload();
        }
        else if (this.honingPoint) {
            const p = new Point2(this.honingPoint.x + HONING_AREA_SIZE * point.x, this.honingPoint.y + HONING_AREA_SIZE * point.y);
            this.honingPoint = null;
            this.addPoint(p);
        }
        else {
            this.hoveringHoningPoint = null;
            this.honingPoint = this.honingPointForCursor(point);
            this.draw();
        }
    }
    handleHover(point) {
        if (this.isLoading) {
            return;
        }
        if (this.img && !this.honingPoint) {
            this.hoveringHoningPoint = this.honingPointForCursor(point);
            this.draw();
        }
    }
    honingPointForCursor(point) {
        const minX = this.offsetX / this.canvas.width;
        const minY = this.offsetY / this.canvas.width;
        const maxX = minX + this.img.width * this.scale / this.canvas.width - HONING_AREA_SIZE;
        const maxY = minY + this.img.height * this.scale / this.canvas.height - HONING_AREA_SIZE;
        return new Point2(Math.min(maxX, Math.max(minX, point.x - HONING_AREA_SIZE / 2)), Math.min(maxY, Math.max(minY, point.y - HONING_AREA_SIZE / 2)));
    }
    addPoint(point) {
        this.points.push(point);
        if (this.points.length === 4) {
            this.gotAllPoints();
        }
        this.draw();
    }
    undo() {
        if (this.honingPoint) {
            this.honingPoint = null;
        }
        else if (this.points.length > 0) {
            this.points.splice(this.points.length - 1, 1);
        }
        this.draw();
    }
    resetImage(img) {
        const iw = img.width;
        const ih = img.height;
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        this.scale = Math.min(cw / iw, ch / ih);
        this.offsetX = Math.round((cw - (this.scale * iw)) / 2);
        this.offsetY = Math.round((ch - (this.scale * ih)) / 2);
        this.img = img;
        this.honingPoint = null;
        this.hoveringHoningPoint = null;
        this.points = [];
        this.isLoading = false;
        this.draw();
    }
    draw() {
        if (!this.isLoading && (this.honingPoint || this.points.length > 0)) {
            this.undoButton.classList.remove("hidden");
        }
        else {
            this.undoButton.classList.add("hidden");
        }
        const ctx = this.canvas.getContext("2d");
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.isLoading) {
            this.drawText(this.loadingStatus);
        }
        else if (!this.img) {
            this.drawText("Drop files or click here");
        }
        else if (!this.honingPoint) {
            ctx.drawImage(this.img, this.offsetX, this.offsetY, this.scale * this.img.width, this.scale * this.img.height);
            if (this.hoveringHoningPoint) {
                ctx.strokeStyle = "green";
                ctx.beginPath();
                ctx.rect(this.hoveringHoningPoint.x * this.canvas.width, this.hoveringHoningPoint.y * this.canvas.width, HONING_AREA_SIZE * this.canvas.width, HONING_AREA_SIZE * this.canvas.height);
                ctx.stroke();
            }
            this.points.forEach((p) => {
                const x = this.canvas.width * p.x;
                const y = this.canvas.height * p.y;
                ctx.fillStyle = "green";
                ctx.beginPath();
                ctx.arc(x, y, 5.0, 0, Math.PI * 2, false);
                ctx.fill();
            });
        }
        else {
            ctx.save();
            ctx.scale(1 / HONING_AREA_SIZE, 1 / HONING_AREA_SIZE);
            ctx.translate(this.offsetX, this.offsetY);
            ctx.translate(-this.honingPoint.x * this.canvas.width, -this.honingPoint.y * this.canvas.height);
            ctx.drawImage(this.img, 0, 0, this.scale * this.img.width, this.scale * this.img.height);
            ctx.restore();
        }
    }
    drawText(text) {
        const ctx = this.canvas.getContext("2d");
        ctx.fillStyle = "#e5e5e5";
        ctx.beginPath();
        ctx.rect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fill();
        ctx.fillStyle = "#777";
        ctx.font = "30px sans-serif";
        ctx.beginPath();
        ctx.textAlign = "center";
        ctx.fillText(text, this.canvas.width / 2, this.canvas.height / 2);
    }
    gotAllPoints() {
        this.isLoading = true;
        this.loadingStatus = "Loading...";
        this.onPick(this.points);
    }
    hide() {
        this.container.classList.add("hidden");
    }
    show() {
        this.container.classList.remove("hidden");
        this.resetImage(this.img);
    }
    showStatus(status) {
        this.loadingStatus = status;
        this.draw();
    }
    maxDimension() {
        return Math.max(this.img.width, this.img.height);
    }
    pixelSource() {
        const scale = Math.max(this.img.width / this.canvas.width, this.img.height / this.canvas.height);
        const extractionCanvas = document.createElement("canvas");
        extractionCanvas.width = this.canvas.width * scale;
        extractionCanvas.height = this.canvas.height * scale;
        const ctx = extractionCanvas.getContext("2d");
        ctx.drawImage(this.img, this.offsetX * scale, this.offsetY * scale, this.scale * this.img.width * scale, this.scale * this.img.height * scale);
        return PixelSource.fromCanvas(extractionCanvas);
    }
}
class FinishDialog {
    constructor(solution, pixelSource, aspectRatio, defaultSize, modelClient) {
        this.solution = solution;
        this.pixelSource = pixelSource;
        this.modelClient = modelClient;
        this.onClose = null;
        this.element = document.getElementById("finish-dialog");
        this.previewContainer = this.element.getElementsByClassName("scaling-preview")[0];
        this.previewContainer.innerHTML = "";
        this.sideLength = this.element.getElementsByClassName("side-length")[0];
        this.aspectRatio = this.element.getElementsByClassName("aspect-ratio")[0];
        this.downloadButton = this.element.getElementsByClassName("download-button")[0];
        this.closeButton = this.element.getElementsByClassName("close-button")[0];
        this.closeButton.addEventListener("click", () => this.onClose());
        this.downloadButton.addEventListener("click", (e) => {
            if (this.downloadButton.textContent === "Prepare") {
                e.preventDefault();
                this.prepareDownload();
            }
            ;
        });
        this.downloadButton.textContent = "Prepare";
        this.aspectRatio.value = Math.log(aspectRatio) + "";
        this.sideLength.value = defaultSize + "";
        this.previewCanvas = document.createElement("canvas");
        this.previewCanvas.width = 256;
        this.previewCanvas.height = 256;
        this.pixelSource.extractProjectedImage(this.solution, this.previewCanvas);
        this.previewCanvas.style.position = "absolute";
        this.updateAspectRatio();
        this.previewContainer.appendChild(this.previewCanvas);
        this.aspectRatio.addEventListener("input", () => this.updateAspectRatio());
    }
    updateAspectRatio() {
        const ratio = Math.exp(parseFloat(this.aspectRatio.value));
        if (ratio > 1) {
            this.previewCanvas.style.width = (100 / ratio).toFixed(2) + "%";
            this.previewCanvas.style.left = (50 - 50 / ratio).toFixed(2) + "%";
            this.previewCanvas.style.height = "100%";
            this.previewCanvas.style.top = "0";
        }
        else {
            this.previewCanvas.style.height = (100 * ratio).toFixed(2) + "%";
            this.previewCanvas.style.top = (50 - 50 * ratio).toFixed(2) + "%";
            this.previewCanvas.style.width = "100%";
            this.previewCanvas.style.left = "0";
        }
    }
    prepareDownload() {
        return __awaiter(this, void 0, void 0, function* () {
            const aspectRatio = Math.exp(parseFloat(this.aspectRatio.value));
            const sideLength = parseInt(this.sideLength.value);
            this.element.classList.add("finish-dialog-disabled");
            try {
                const downloadURL = yield this.modelClient.exportImage(this.solution, this.pixelSource, aspectRatio, sideLength);
                this.downloadButton.href = downloadURL;
                this.downloadButton.textContent = "Download";
                this.downloadButton.download = ("flattened_" + Math.round(new Date().getTime() / 1000) + ".png");
            }
            finally {
                this.element.classList.remove("finish-dialog-disabled");
            }
        });
    }
    show() {
        this.element.style.display = "block";
    }
    hide() {
        this.element.style.display = "none";
    }
}
function canvasToTensor(canvas) {
    const imgData = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
    const result = Tensor.zeros(Shape.make(1, 3, canvas.height, canvas.width));
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            for (let k = 0; k < 3; k++) {
                const data = imgData.data[(y * canvas.width + x) * 4 + k] / 255;
                result.data[(k * canvas.height + y) * canvas.width + x] = data;
            }
        }
    }
    return result;
}
class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}
//# sourceMappingURL=app.js.map
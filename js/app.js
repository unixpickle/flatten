class App {
    constructor() {
        this.worker = new Worker('js/worker.js');
        this.worker.onmessage = (e) => {
            const msg = e.data;
            console.log(msg);
            if (msg["error"]) {
                alert("Error: " + msg.error);
            } else {
                const solVec = nn.Tensor.fromData(msg.solution);
                const solution = nn.PerspectiveSolution.fromFlatVec(solVec);
                this.handleSolution(solution);
            }
        };

        this.dropZone = document.getElementById('drop-zone');
        this.canvas = document.getElementById('picker');

        this.dropZone.addEventListener('dragover', (e) => this.handleDragOver(e), false);
        this.dropZone.addEventListener('drop', (e) => this.handleFileSelect(e), false);

        this.canvas.addEventListener('mousedown', (e) => {
            if (this._points.length === 4) {
                this._points = [];
            }
            const x = e.offsetX;
            const y = e.offsetY;
            this._points.push(new Point2(
                x / this.canvas.width,
                y / this.canvas.height,
            ));
            this.draw();
            if (this._points.length === 4) {
                this.solve();
            }
        });

        // Properties for rendering the image.
        this._img = null;
        this._scale = null;
        this._offsetX = 0;
        this._offsetY = 0;

        // Points for fitting.
        this._points = [];
    }

    handleDragOver(evt) {
        evt.stopPropagation();
        evt.preventDefault();
        evt.dataTransfer.dropEffect = 'copy';
    }

    handleFileSelect(evt) {
        evt.stopPropagation();
        evt.preventDefault();

        var files = evt.dataTransfer.files;
        for (let i = 0, f; f = files[i]; i++) {
            if (!f.type.match('image.*')) {
                continue;
            }

            var reader = new FileReader();

            reader.onload = (e) => {
                var img = new Image();
                img.onload = () => {
                    const iw = img.width;
                    const ih = img.height;
                    const cw = this.canvas.width;
                    const ch = this.canvas.height;
                    this._scale = Math.min(cw / iw, ch / ih);
                    this._offsetX = Math.round(cw - (this._scale * iw));
                    this._offsetY = Math.round(ch - (this._scale * ih));
                    this._img = img;
                    this.draw();
                };
                img.src = e.target.result;

                this.dropZone.style.display = 'none';
                this.canvas.style.display = 'block';
            };

            reader.readAsDataURL(f);
        }
    }

    draw() {
        const ctx = this.canvas.getContext('2d');
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.drawImage(
            this._img,
            this._offsetX,
            this._offsetY,
            this._scale * this._img.width,
            this._scale * this._img.height,
        );
        this._points.forEach((p) => {
            const x = this.canvas.width * p.x;
            const y = this.canvas.height * p.y;
            ctx.fillStyle = 'green';
            ctx.beginPath();
            ctx.arc(x, y, 5.0, 0, Math.PI * 2, false);
            ctx.fill();
        });
    }

    solve() {
        this.worker.postMessage({
            points: this._points.map((p) => [p.x, p.y])
        });
    }

    handleSolution(solution) {
        const source = this.pixelSource();
        const projector = solution.projector();

        const [w, h] = solution.size.toList();
        const scale = Math.min(200 / w, 200 / h);

        const dstCanvas = document.createElement('canvas');
        dstCanvas.width = Math.ceil(w * scale);
        dstCanvas.height = Math.ceil(h * scale);
        const dst = dstCanvas.getContext('2d');
        const imgData = dst.createImageData(dstCanvas.width, dstCanvas.height);

        for (let x = 0; x < dstCanvas.width; x++) {
            for (let y = 0; y < dstCanvas.height; y++) {
                const dstPoint = new Point2(x / scale, y / scale);
                const sourcePoint = projector(dstPoint);
                const sourcePixel = source(sourcePoint.x, sourcePoint.y);
                const idx = (x + y * dstCanvas.width) * 4;
                for (let i = 0; i < 4; i++) {
                    imgData.data[idx + i] = sourcePixel[i];
                }
            }
        }

        dst.putImageData(imgData, 0, 0);
        document.body.appendChild(dstCanvas);
    }

    pixelSource() {
        const scale = Math.max(
            this._img.width / this.canvas.width,
            this._img.height / this.canvas.height,
        );
        const extractionCanvas = document.createElement('canvas');
        extractionCanvas.width = this.canvas.width * scale;
        extractionCanvas.height = this.canvas.height * scale;
        const ctx = extractionCanvas.getContext('2d');
        ctx.drawImage(
            this._img,
            this._offsetX * scale,
            this._offsetY * scale,
            this._scale * this._img.width * scale,
            this._scale * this._img.height * scale,
        );

        const data = ctx.getImageData(0, 0, extractionCanvas.width, extractionCanvas.height);
        return (relX, relY) => {
            const x = Math.round(Math.max(0, Math.min(1, relX)) * extractionCanvas.width);
            const y = Math.round(Math.max(0, Math.min(1, relY)) * extractionCanvas.height);
            const index = (x + y * extractionCanvas.width) * 4;
            return data.data.slice(index, index + 4);
        };
    }
}

class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

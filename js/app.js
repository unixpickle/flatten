class App {
    constructor() {
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
            console.log(this._points);
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
        const solver = new Solver(this._points);
        for (let i = 0; i < 2000; i++) {
            solver.step();
        }
        const source = this.pixelSource();
        const map = solver.solutionMap();

        const w = map.width;
        const h = map.height;
        const scale = Math.min(64 / w, 64 / h);

        const dstCanvas = document.createElement('canvas');
        dstCanvas.width = Math.ceil(w * scale);
        dstCanvas.height = Math.ceil(h * scale);
        const dst = dstCanvas.getContext('2d');
        const imgData = dst.createImageData(dstCanvas.width, dstCanvas.height);

        for (let sx = 0; sx < 1; sx += 0.001) {
            for (let sy = 0; sy < 1; sy += 0.001) {
                const sourcePixel = source(sx, sy);
                const dest = map.destPoint(new Point2(sx, sy));
                const x = Math.round(dest.x * scale);
                const y = Math.round(dest.y * scale);
                if (x >= 0 && x < dstCanvas.width && y >= 0 && y < dstCanvas.height) {
                    const idx = (x + y * dstCanvas.width) * 4;
                    for (let i = 0; i < 4; i++) {
                        imgData.data[idx + i] = sourcePixel[i];
                    }
                }
            }
        }

        dst.putImageData(imgData, 0, 0);
        document.body.appendChild(dstCanvas);
    }

    pixelSource() {
        const extractionCanvas = document.createElement('canvas');
        extractionCanvas.width = this.canvas.width;
        extractionCanvas.height = this.canvas.height;
        const ctx = extractionCanvas.getContext('2d');
        ctx.drawImage(
            this._img,
            this._offsetX,
            this._offsetY,
            this._scale * this._img.width,
            this._scale * this._img.height,
        );

        const data = ctx.getImageData(0, 0, extractionCanvas.width, extractionCanvas.height);
        return (relX, relY) => {
            const x = Math.round(relX * extractionCanvas.width);
            const y = Math.round(relY * extractionCanvas.height);
            const index = (x + y * extractionCanvas.width) * 4;
            return data.data.slice(index, index + 4);
        };
    }
}
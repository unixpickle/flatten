class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    dist(other) {
        return Math.sqrt(Math.pow(other.x - this.x, 2) + Math.pow(other.y - this.y, 2));
    }
}

class Solution {
    constructor(rotX, rotY, rotZ, camX, camY, camZ, width, height) {
        this.rotX = rotX;
        this.rotY = rotY;
        this.rotZ = rotZ;
        this.camX = camX;
        this.camY = camY;
        this.camZ = camZ;
        this.width = width;
        this.height = height;
    }

    addScaled(gradient, scale) {
        return new Solution(
            this.rotX + scale * gradient.rotX,
            this.rotY + scale * gradient.rotY,
            this.rotZ + scale * gradient.rotZ,
            this.camX + scale * gradient.camX,
            this.camY + scale * gradient.camY,
            this.camZ + scale * gradient.camZ,
            this.width + scale * gradient.width,
            this.height + scale * gradient.height,
        );
    }
}

class Camera {
    constructor(rotation, origin) {
        this.rotation = rotation;
        this.origin = origin;
    }

    project(x) {
        const point = this.origin.sub(x);
        const rotated = this.rotation.mulVec(point);
        const scale = rotated.z.pow(-1);
        return new Vector2(
            rotated.x.mul(scale),
            rotated.y.mul(scale),
        );
    }

    constant(x) {
        return this.rotation.a.constant(x);
    }
}

class SolutionMap {
    constructor(camera, points) {
        const v1 = points[1].sub(points[0]);
        const v2 = points[3].sub(points[0]);

        this._camera = camera;
        this._origin = points[0];
        this._x = v1.normalize();
        this._y = v2.normalize();
        this.width = v1.norm().value;
        this.height = v2.norm().value;
    }

    constant(x) {
        return this._x.x.constant(x);
    }

    source(dst) {
        const xOff = this._x.scale(this.constant(dst.x));
        const yOff = this._y.scale(this.constant(dst.y));
        const p = this._origin.add(xOff).add(yOff)
        const proj = this._camera.project(p);
        return new Point2(proj.x.value, proj.y.value);
    }
}

class Solver {
    constructor(corners) {
        this.corners = corners;
        this.solution = new Solution(
            // Rotations
            0, 0, 0,
            // Origin
            0, 0, -1.0,
            // Width and height
            corners[1].dist(corners[0]),
            corners[3].dist(corners[0]),
        );
    }

    step() {
        const [gradient, loss] = this.gradient();
        this.solution = this.solution.addScaled(gradient, -0.01);
        return loss;
    }

    solutionMap() {
        const [camera, points] = this.rawCameraAndPoints();
        return new SolutionMap(camera, points);
    }

    gradient() {
        const [camera, points] = this.cameraAndPoints();
        const projPoints = points.map((p) => camera.project(p));

        let projMSE = camera.constant(0);
        projPoints.forEach((x, i) => {
            const target = new Vector2(
                camera.constant(this.corners[i].x),
                camera.constant(this.corners[i].y),
            );
            const diff = x.sub(target);
            projMSE = projMSE.add(diff.dot(diff));
        });

        return [new Solution(...projMSE.derivatives), projMSE.value];
    }

    cameraAndPoints() {
        return this._cameraAndPoints(this._vars());
    }

    rawCameraAndPoints() {
        return this._cameraAndPoints(this._vars().map((x) => new DualNumber(x.value, [])));
    }

    _vars() {
        return DualNumber.variables(
            this.solution.rotX,
            this.solution.rotY,
            this.solution.rotZ,
            this.solution.camX,
            this.solution.camY,
            this.solution.camZ,
            this.solution.width,
            this.solution.height,
        );
    }

    _cameraAndPoints(vars) {
        const [rotX, rotY, rotZ, camX, camY, camZ, width, height] = vars;
        const rotation = Matrix3.eulerRotation(rotX, rotY, rotZ);
        const camOrigin = new Vector3(camX, camY, camZ);
        const zero = rotX.constant(0);
        const x1 = rotX.constant(this.corners[0].x);
        const y1 = rotX.constant(this.corners[0].y);
        return [
            new Camera(rotation, camOrigin),
            [
                new Vector3(x1, y1, zero),
                new Vector3(x1.add(width), y1, zero),
                new Vector3(x1.add(width), y1.add(height), zero),
                new Vector3(x1, y1.add(height), zero),
            ],
        ];
    }
}
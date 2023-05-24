class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

class Solution {
    constructor(rotX, rotY, rotZ, camX, camY, camZ, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y) {
        this.rotX = rotX;
        this.rotY = rotY;
        this.rotZ = rotZ;
        this.camX = camX;
        this.camY = camY;
        this.camZ = camZ;
        this.p1x = p1x;
        this.p1y = p1y;
        this.p2x = p2x;
        this.p2y = p2y;
        this.p3x = p3x;
        this.p3y = p3y;
        this.p4x = p4x;
        this.p4y = p4y;
    }

    addScaled(gradient, scale) {
        return new Solution(
            this.rotX + scale * gradient.rotX,
            this.rotY + scale * gradient.rotY,
            this.rotZ + scale * gradient.rotZ,
            this.camX + scale * gradient.camX,
            this.camY + scale * gradient.camY,
            this.camZ + scale * gradient.camZ,
            this.p1x + scale * gradient.p1x,
            this.p1y + scale * gradient.p1y,
            this.p2x + scale * gradient.p2x,
            this.p2y + scale * gradient.p2y,
            this.p3x + scale * gradient.p3x,
            this.p3y + scale * gradient.p3y,
            this.p4x + scale * gradient.p4x,
            this.p4y + scale * gradient.p4y,
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
            0, 0, 0, 0, 0, -1.0,
            corners[0].x,
            corners[0].y,
            corners[1].x,
            corners[1].y,
            corners[2].x,
            corners[2].y,
            corners[3].x,
            corners[3].y,
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

        const cornerVecs = points.map((c1, i) => {
            const c2 = points[(i + 1) % points.length];
            return c2.sub(c1).normalize();
        });

        let sqDotSum = camera.constant(0);
        for (let i = 0; i < cornerVecs.length; i++) {
            const v1 = cornerVecs[i]
            const v2 = cornerVecs[(i + 1) % cornerVecs.length];
            sqDotSum = sqDotSum.add(v1.dot(v2).pow(2));
        }

        let loss = sqDotSum.scale(0.01).add(projMSE);

        return [new Solution(...loss.derivatives), loss];
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
            this.solution.p1x,
            this.solution.p1y,
            this.solution.p2x,
            this.solution.p2y,
            this.solution.p3x,
            this.solution.p3y,
            this.solution.p4x,
            this.solution.p4y,
        );
    }

    _cameraAndPoints(vars) {
        const [rotX, rotY, rotZ, camX, camY, camZ, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y] = vars;
        const rotation = Matrix3.eulerRotation(rotX, rotY, rotZ);
        const camOrigin = new Vector3(camX, camY, camZ);
        const zero = p1x.constant(0);
        return [
            new Camera(rotation, camOrigin),
            [
                new Vector3(p1x, p1y, zero),
                new Vector3(p2x, p2y, zero),
                new Vector3(p3x, p3y, zero),
                new Vector3(p4x, p4y, zero),
            ],
        ];
    }
}
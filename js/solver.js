class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

class Solution {
    constructor(rotX, rotY, rotZ, camX, camY, camZ) {
        this.rotX = rotX;
        this.rotY = rotY;
        this.rotZ = rotZ;
        this.camX = camX;
        this.camY = camY;
        this.camZ = camZ;
    }

    addScaled(gradient, scale) {
        return new Solution(
            this.rotX + scale * gradient.rotX,
            this.rotY + scale * gradient.rotY,
            this.rotZ + scale * gradient.rotZ,
            this.camX + scale * gradient.camX,
            this.camY + scale * gradient.camY,
            this.camZ + scale * gradient.camZ,
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
    constructor(camera, projCorners) {
        const v1 = projCorners[1].sub(projCorners[0]);
        const v2 = projCorners[3].sub(projCorners[0]);
        this._camera = camera;
        this._origin = projCorners[0];
        this._x = v1.normalize();
        this._y = v2.normalize();
        this.width = v1.norm().value;
        this.height = v2.norm().value;
    }

    destPoint(source) {
        const proj = this._camera.project(
            new Vector3(
                this._camera.constant(source.x),
                this._camera.constant(source.y),
                this._camera.constant(0),
            ),
        ).sub(this._origin);
        return new Point2(
            this._x.dot(proj).value,
            this._y.dot(proj).value,
        );
    }
}

class Solver {
    constructor(corners) {
        this.corners = corners;
        this.solution = new Solution(0, 0, 0, 0, 0, 1.0);
    }

    step() {
        const [gradient, loss] = this.gradient();
        this.solution = this.solution.addScaled(gradient, -0.01);
        return loss;
    }

    solutionMap() {
        const camera = this.rawCamera();
        return new SolutionMap(
            camera,
            this._projCorners(camera),
        );
    }

    gradient() {
        const camera = this.camera();

        const projCorners = this._projCorners(camera);

        const cornerVecs = projCorners.map((c1, i) => {
            const c2 = projCorners[(i + 1) % projCorners.length];
            return c2.sub(c1).normalize();
        });

        let sqDotSum = camera.constant(0);
        for (let i = 0; i < cornerVecs.length; i++) {
            const v1 = cornerVecs[i]
            const v2 = cornerVecs[(i + 1) % cornerVecs.length];
            sqDotSum = sqDotSum.add(v1.dot(v2).pow(2));
        }

        return [new Solution(...sqDotSum.derivatives), sqDotSum.value];
    }

    _projCorners(camera) {
        return this.corners.map((c) => {
            return camera.project(
                new Vector3(
                    camera.constant(c.x),
                    camera.constant(c.y),
                    camera.constant(0),
                ),
            );
        })
    }

    camera() {
        return this._camera(this._vars());
    }

    rawCamera() {
        return this._camera(this._vars().map((x) => new DualNumber(x.value, [])));
    }

    _vars() {
        return DualNumber.variables(
            this.solution.rotX,
            this.solution.rotY,
            this.solution.rotZ,
            this.solution.camX,
            this.solution.camY,
            this.solution.camZ,
        );
    }

    _camera(vars) {
        const [rotX, rotY, rotZ, camX, camY, camZ] = vars;
        const rotation = Matrix3.eulerRotation(rotX, rotY, rotZ);
        const camOrigin = new Vector3(camX, camY, camZ);
        return new Camera(rotation, camOrigin);
    }
}
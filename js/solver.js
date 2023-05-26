class Point2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    add(other) {
        return new Point2(this.x + other.x, this.y + other.y);
    }

    sub(other) {
        return new Point2(this.x - other.x, this.y - other.y);
    }

    scale(s) {
        return new Point2(this.x * s, this.y * s);
    }

    norm() {
        return Math.sqrt(Math.pow(this.x, 2) + Math.pow(this.y, 2));
    }

    normalize() {
        return this.scale(1 / this.norm());
    }

    dot(other) {
        return this.x * other.x + this.y * other.y;
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

    sign() {
        return new Solution(
            Math.sign(this.rotX),
            Math.sign(this.rotY),
            Math.sign(this.rotZ),
            Math.sign(this.camX),
            Math.sign(this.camY),
            Math.sign(this.camZ),
            Math.sign(this.width),
            Math.sign(this.height),
        )
    }
}

class Camera {
    constructor(rotation, origin) {
        this.rotation = rotation;
        this.origin = origin;
    }

    project(x) {
        const point = this.rotation.mulVec(x).add(this.origin);
        const scale = point.z.pow(-1);
        return new Vector2(
            point.x.mul(scale),
            point.y.mul(scale),
        );
    }

    constant(x) {
        return this.rotation.a.constant(x);
    }
}

class SolutionMap {
    constructor(camera, points, cornerOffset) {
        const v1 = points[1].sub(points[0]);
        const v2 = points[3].sub(points[0]);

        this._camera = camera;
        this._origin = points[0];
        this._x = v1.normalize();
        this._y = v2.normalize();
        this._cornerOffset = cornerOffset;
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
        const proj = this._camera.project(p).add(this._cornerOffset);
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
        this.targetDots = cornerDots(corners).map((x) => Math.abs(x));
        this.steps = 0;
    }

    initSearch(n) {
        let loss = this.loss(false).value;
        const stops = [];
        for (let i = 0; i < n; i++) {
            stops.push(2.0 / (n - 1) - 1);
        }
        stops.forEach((rotX) => {
            stops.forEach((rotY) => {
                stops.forEach((rotZ) => {
                    stops.forEach((zOff) => {
                        stops.forEach((scaleStop) => {
                            const z = zOff - 1.1; // Should be <= -0.1
                            const scale = (scaleStop + 1.5); // Should be in [0.5, 2.5].
                            const s1 = new Solution(
                                rotX * 0.2,
                                rotY * 0.2,
                                rotZ * 0.2,
                                0,
                                0,
                                z,
                                this.solution.width * scale,
                                this.solution.height * scale,
                            );
                            const oldSolution = this.solution;
                            const oldStep = this.step;
                            this.solution = s1;
                            this.steps = 0;
                            let newLoss;
                            for (let i = 0; i < 2; i++) {
                                newLoss = this.step();
                            }
                            if (newLoss < loss) {
                                loss = newLoss;
                            } else {
                                this.solution = oldSolution;
                                this.step = oldStep;
                            }
                        });
                    });
                });
            });
        });
        return loss;
    }

    step() {
        const lr = 0.01 * Math.max(0, 5000 - this.steps) / 5000 + 0.01;
        this.steps += 1;
        const [gradient, loss] = this.gradient();
        this.solution = this.solution.addScaled(gradient, -lr);
        return loss;
    }

    solutionMap() {
        const [camera, points] = this.cameraAndPoints(false);
        const cornerFirst = new Vector2(
            camera.constant(this.corners[0].x),
            camera.constant(this.corners[0].y),
        );
        const projFirst = camera.project(points[0]);
        const cornerOffset = cornerFirst.sub(projFirst);
        return new SolutionMap(camera, points, cornerOffset);
    }

    gradient() {
        const loss = this.loss(true);
        return [new Solution(...loss.derivatives), loss.value];
    }

    loss(derivatives) {
        const [camera, points] = this.cameraAndPoints(derivatives);
        const projPoints = points.map((p) => camera.project(p));

        const cornerTargets = this.corners.map((c) => new Vector2(
            camera.constant(c.x),
            camera.constant(c.y),
        ));

        // Ignore the exact canvas position to get better MSE.
        const cornerOffset = cornerTargets[0].sub(projPoints[0]);

        let projMSE = camera.constant(0);
        projPoints.forEach((x, i) => {
            const diff = x.add(cornerOffset).sub(cornerTargets[i]);
            projMSE = projMSE.add(diff.dot(diff));
        });

        const targetDots = this.targetDots.map((x) => camera.constant(x));
        const sourceDots = cornerDots(projPoints);
        let dotMSE = camera.constant(0);
        sourceDots.forEach((x, i) => {
            dotMSE = dotMSE.add(x.abs().sub(targetDots[i]).pow(2));
        });

        // Trace is equal to 1+2*cos(theta), so maximizing it
        // reduces the rotation angle.
        const rotTerm = camera.rotation.trace().scale(0.0);// .scale(-0.0001);

        return projMSE.add(dotMSE).add(rotTerm);
    }

    cameraAndPoints(derivatives) {
        let vars = this._vars();
        if (!derivatives) {
            vars = vars.map((x) => new DualNumber(x.value, []));
        }
        return this._cameraAndPoints(vars);
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

function cornerDots(points) {
    const deltas = points.map((x, i) => points[(i + 1) % points.length].sub(x).normalize());
    return deltas.map((x, i) => x.dot(deltas[(i + 1) % deltas.length]));
}

function testExample() {
    const zero = new DualNumber(0, []);
    const camera = new Camera(
        Matrix3.eulerRotation(zero, zero.constant(0.1), zero),
        new Vector3(zero, zero, zero.constant(-1)),
    );
    const corners = [
        new Point2(0.3, 0.3),
        new Point2(0.45, 0.3),
        new Point2(0.45, 0.45),
        new Point2(0.3, 0.45),
    ];

    let projCorners = corners.map((c) => {
        const v = new Vector3(zero.constant(c.x), zero.constant(c.y), zero);
        const proj = camera.project(v);
        return new Point2(proj.x.value, proj.y.value);
    });
    const offset = corners[0].sub(projCorners[0]);
    projCorners = projCorners.map((x) => x.add(offset));

    const solver = new Solver(projCorners);
    solver.initSearch(5);
    // solver.solution.rotY = 0.1;
    // solver.solution.width = 0.15;
    // solver.solution.height = 0.15;
    console.log(solver.step());
    for (let i = 0; i < 10000; i++) {
        solver.step();
    }
    console.log(solver.step());
    console.log(solver.solution);
}

testExample();
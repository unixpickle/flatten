class DualNumber {
    constructor(value, derivatives) {
        this.value = value;
        this.derivatives = derivatives;
    }

    static variables(...values) {
        return values.map((val, i) => {
            const derivatives = [];
            for (let j = 0; j < values.length; j++) {
                derivatives.push(i == j ? 1 : 0);
            }
            return new DualNumber(val, derivatives);
        });
    }

    add(other) {
        return new DualNumber(
            this.value + other.value,
            this.derivatives.map((x, i) => x + other.derivatives[i]),
        )
    }

    sub(other) {
        return new DualNumber(
            this.value - other.value,
            this.derivatives.map((x, i) => x - other.derivatives[i]),
        )
    }

    mul(other) {
        return new DualNumber(
            this.value * other.value,
            this.derivatives.map((x, i) => x * other.value + other.derivatives[i] * this.value),
        )
    }

    div(other) {
        return this.mul(other.pow(-1));
    }

    pow(p) {
        return new DualNumber(
            Math.pow(this.value, p),
            this.derivatives.map((x) => x * p * Math.pow(this.value, p - 1)),
        )
    }

    addScalar(s) {
        return new DualNumber(
            this.value + s,
            this.derivatives,
        )
    }

    scale(s) {
        return new DualNumber(
            this.value * s,
            this.derivatives.map((x) => x * s),
        )
    }

    cos() {
        return new DualNumber(
            Math.cos(this.value),
            this.derivatives.map((x) => -x * Math.sin(this.value)),
        )
    }

    sin() {
        return new DualNumber(
            Math.sin(this.value),
            this.derivatives.map((x) => x * Math.cos(this.value)),
        )
    }

    constant(x) {
        return new DualNumber(
            x,
            this.derivatives.map((_) => 0),
        );
    }
}

class Vector2 {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    scale(x) {
        return new Vector2(this.x.mul(x), this.y.mul(x));
    }

    add(other) {
        return new Vector2(
            this.x.add(other.x),
            this.y.add(other.y),
        )
    }

    sub(other) {
        return new Vector2(
            this.x.sub(other.x),
            this.y.sub(other.y),
        )
    }

    dot(other) {
        return this.x.mul(other.x).add(this.y.mul(other.y));
    }

    norm() {
        return this.x.pow(2).add(this.y.pow(2)).pow(0.5);
    }

    normalize() {
        return this.scale(this.norm().pow(-1));
    }
}

class Vector3 {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    scale(x) {
        return new Vector3(this.x.mul(x), this.y.mul(x), this.z.mul(x));
    }

    add(other) {
        return new Vector3(
            this.x.add(other.x),
            this.y.add(other.y),
            this.z.add(other.z),
        )
    }

    sub(other) {
        return new Vector3(
            this.x.sub(other.x),
            this.y.sub(other.y),
            this.z.sub(other.z),
        )
    }

    dot(other) {
        return this.x.mul(other.x).add(this.y.mul(other.y)).add(this.z.mul(other.z));
    }

    norm() {
        return this.x.pow(2).add(this.y.pow(2)).add(this.z.pow(2)).pow(0.5);
    }

    normalize() {
        return this.scale(this.norm().pow(-1));
    }
}

class Matrix3 {
    constructor(a, b, c, d, e, f, g, h, i) {
        this.a = a;
        this.b = b;
        this.c = c;
        this.d = d;
        this.e = e;
        this.f = f;
        this.g = g;
        this.h = h;
        this.i = i;
    }

    static eulerRotation(x, y, z) {
        return Matrix3.rotationZ(z).mul(Matrix3.rotationY(y)).mul(Matrix3.rotationX(x));
    }

    static rotationX(theta) {
        const cos = theta.cos();
        const sin = theta.sin();
        const zero = theta.constant(0);
        const one = theta.constant(1);
        return new Matrix3(
            one, zero, zero,
            zero, cos, sin.scale(-1),
            zero, sin, cos,
        );
    }

    static rotationY(theta) {
        const cos = theta.cos();
        const sin = theta.sin();
        const zero = theta.constant(0);
        const one = theta.constant(1);
        return new Matrix3(
            cos, zero, sin,
            zero, one, zero,
            sin.scale(-1), zero, cos,
        );
    }

    static rotationZ(theta) {
        const cos = theta.cos();
        const sin = theta.sin();
        const zero = theta.constant(0);
        const one = theta.constant(1);
        return new Matrix3(
            cos, sin.scale(-1), zero,
            sin, cos, zero,
            zero, zero, one,
        );
    }

    mul(other) {
        return new Matrix3(
            this.a.mul(other.a).add(this.b.mul(other.d)).add(this.c.mul(other.g)),
            this.a.mul(other.b).add(this.b.mul(other.e)).add(this.c.mul(other.h)),
            this.a.mul(other.c).add(this.b.mul(other.f)).add(this.c.mul(other.i)),
            this.d.mul(other.a).add(this.e.mul(other.d)).add(this.f.mul(other.g)),
            this.d.mul(other.b).add(this.e.mul(other.e)).add(this.f.mul(other.h)),
            this.d.mul(other.c).add(this.e.mul(other.f)).add(this.f.mul(other.i)),
            this.g.mul(other.a).add(this.h.mul(other.d)).add(this.i.mul(other.g)),
            this.g.mul(other.b).add(this.h.mul(other.e)).add(this.i.mul(other.h)),
            this.g.mul(other.c).add(this.h.mul(other.f)).add(this.i.mul(other.i))
        );
    }

    mulVec(vec) {
        return new Vector3(
            this.a.mul(vec.x).add(this.b.mul(vec.y)).add(this.c.mul(vec.z)),
            this.d.mul(vec.x).add(this.e.mul(vec.y)).add(this.f.mul(vec.z)),
            this.g.mul(vec.x).add(this.h.mul(vec.y)).add(this.i.mul(vec.z)),
        );
    }
}

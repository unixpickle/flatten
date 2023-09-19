function testTranspose() {
    const data = Tensor.fromData([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [-3, -5, -7],
            [-4, -2, -1],
        ]
    ]);
    const out = data.t();
    const expected = Tensor.fromData([
        [
            [1, 4],
            [2, 5],
            [3, 6],
        ],
        [
            [-3, -4],
            [-5, -2],
            [-7, -1],
        ]
    ]);
    assertEqual(out, expected);

    console.log('[Done] transpose');
}

function testLinear() {
    const weights = Tensor.fromData([
        [1, 2, 3],
        [4, 5, 6],
    ]);
    let weightGrad: Tensor = null;
    weights.backward = (x) => weightGrad = x;

    const biases = Tensor.fromData([-1, -3, -2]);
    let biasGrad: Tensor = null;
    biases.backward = (x) => biasGrad = x;

    const inputs = Tensor.fromData([[2, 4]]);
    let inputGrad: Tensor = null;
    inputs.backward = (x) => inputGrad = x;

    const layer = new Linear(weights, biases);
    const output = layer.forward(inputs);

    console.assert(output.shape.equals(Shape.make(1, 3)));
    console.assert(output.data[0] === 17);
    console.assert(output.data[1] === 21);
    console.assert(output.data[2] === 28);

    const loss = output.sum();
    console.assert(loss.data[0] === 66);

    loss.backward(Tensor.fromData(1));

    console.assert(weightGrad.shape.equals(weights.shape));
    console.assert(biasGrad.shape.equals(biases.shape));
    console.assert(inputGrad.shape.equals(inputs.shape));

    const wGrad = new Float32Array([2, 2, 2, 4, 4, 4]);
    const bGrad = new Float32Array([1, 1, 1]);
    const inGrad = new Float32Array([6, 15]);
    weightGrad.data.forEach((x, i) => {
        console.assert(wGrad[i] === x);
    });
    biasGrad.data.forEach((x, i) => {
        console.assert(bGrad[i] === x);
    });
    inputGrad.data.forEach((x, i) => {
        console.assert(inGrad[i] === x);
    });
    console.log('[Done] Linear');
}

function testSlice() {
    const x = Tensor.fromData([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
        [
            [3, 3, 5],
            [5, -10, -20],
        ],
    ]);
    let xGrad = null;
    x.backward = (g) => xGrad = g;
    console.assert(x.shape.equals(Shape.make(4, 2, 3)));

    const chunk = x.slice(0, 1, 3);
    assertEqual(chunk, Tensor.fromData([
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
    ]));

    const chunk1 = x.slice(2, 1, 3);
    chunk1.backward(chunk1.detach().scale(-1));
    assertEqual(xGrad, Tensor.fromData([
        [
            [0, 2, 3],
            [0, 5, 6],
        ],
        [
            [0, 8, 9],
            [0, 11, 12],
        ],
        [
            [0, -2, -3],
            [0, -5, -6],
        ],
        [
            [0, 3, 5],
            [0, -10, -20],
        ],
    ]).scale(-1));

    console.log('[Done] slice');
}

function testAccumGrad() {
    const x = Tensor.fromData([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
        [
            [3, 3, 5],
            [5, -10, -20],
        ],
    ]);
    let xGrad = null;
    x.backward = (g) => xGrad = g;

    const combined = x.accumGrad((x) => {
        const chunk1 = x.slice(2, 1, 3);
        const chunk2 = x.slice(2, 0, 1);
        return Tensor.cat([chunk2, chunk1], 2);
    });
    combined.backward(x.scale(-3));
    assertEqual(xGrad, x.scale(-3));

    console.log('[Done] accumGrad');
}

function testCat() {
    const x = Tensor.fromData([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
        [
            [3, 3, 5],
            [5, -10, -20],
        ],
    ]);

    const t1 = Tensor.cat([x.slice(0, 0, 2), x.slice(0, 2, 3), x.slice(0, 3, 4)], 0);
    assertEqual(x, t1);

    const t2 = Tensor.cat([x.slice(1, 0, 1), x.slice(1, 1)], 1);
    assertEqual(x, t2);

    // Test zero-sized tensor.
    const t3 = Tensor.cat([x.slice(1, 0, 2), x.slice(1, 2, 2)], 1);
    assertEqual(x, t3);

    const t4 = Tensor.cat([x.slice(2, 0, 2), x.slice(2, 2, 3)], 2);
    assertEqual(x, t4);

    const x1 = Tensor.fromData([[[1], [2]], [[4], [5]]]);
    const x2 = Tensor.fromData([[[-2]], [[-1]]]);
    let x1Grad = null;
    let x2Grad = null;
    x1.backward = (x) => x1Grad = x;
    x2.backward = (x) => x2Grad = x;

    const joined = Tensor.cat([x1, x2], 1);
    const grad = Tensor.fromData([[[-3], [-4], [-8]], [[-5], [-6], [-9]]]);
    joined.backward(grad);

    assertEqual(x1Grad, Tensor.fromData([[[-3], [-4]], [[-5], [-6]]]));
    assertEqual(x2Grad, Tensor.fromData([[[-8]], [[-9]]]));

    console.log('[Done] cat');
}

function testSinCos() {
    const t = Tensor.fromData([1, 2, 3, 4]);
    let tGrad: Tensor = null;
    t.backward = (g) => tGrad = g;

    t.sin().sum().backward(Tensor.fromData(1));
    tGrad.data.forEach((x, i) => {
        const y = Math.cos(t.data[i]);
        console.assert(Math.abs(x - y) < 1e-5, x, y);
    });

    t.cos().sum().backward(Tensor.fromData(1));
    tGrad.data.forEach((x, i) => {
        const y = -Math.sin(t.data[i]);
        console.assert(Math.abs(x - y) < 1e-5, x, y);
    });

    console.log('[Done] sin/cos');
}

function testPow() {
    const x = Tensor.fromData([1, 2, 3]);
    let xGrad = null;
    x.backward = (g) => xGrad = g;

    const out = x.pow(3);
    assertEqual(out, Tensor.fromData([1, 2 * 2 * 2, 3 * 3 * 3]));

    out.sum().backward(Tensor.fromData(1));
    assertEqual(xGrad, Tensor.fromData([3, 3 * 2 * 2, 3 * 3 * 3]));

    console.log('[Done] pow');
}

function testExp() {
    const x = Tensor.fromData([1, 2, 3, -1]);
    let xGrad: Tensor = null;
    x.backward = (g) => xGrad = g;

    const out = x.exp();
    out.data.forEach((y, i) => {
        console.assert(Math.abs(y - Math.exp(x.data[i])) < 1e-5, y, x.data[i]);
    });

    out.sum().backward(Tensor.fromData(-3));
    xGrad.data.forEach((g, i) => {
        console.assert(Math.abs(g - -3 * out.data[i]) < 1e-5, g, x.data[i]);
    });

    console.log('[Done] exp')
}

function testReLU() {
    const x = Tensor.fromData([1, 2, -4, 0, 3, -1]);
    let xGrad = null;
    x.backward = (g) => xGrad = g;

    const out = x.relu();
    assertEqual(out, Tensor.fromData([1, 2, 0, 0, 3, 0]));

    out.sum().backward(Tensor.fromData(-3));
    assertEqual(xGrad, Tensor.fromData([-3, -3, 0, 0, -3, 0]));

    console.log('[Done] ReLU');
}

function testRotation() {
    for (let axis = 0; axis < 3; ++axis) {
        [-0.3, 0.3, Math.PI, Math.PI * 2].forEach((theta) => {
            const tensor = rotation(axis, Tensor.fromData(theta));

            // Make sure it's orthonormal.
            const product = matmul(tensor, tensor.t());
            const identity = Tensor.fromData([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
            identity.data.forEach((x, i) => {
                console.assert(Math.abs(x - product.data[i]) < 1e-5, product)
            });

            // Make sure negative angle is the inverse.
            const invRot = rotation(axis, Tensor.fromData(-theta));
            invRot.t().data.forEach((x, i) => {
                console.assert(Math.abs(x - tensor.data[i]) < 1e-5, tensor, invRot);
            });

            // Test backward pass with finite differences.
            const thetaT = Tensor.fromData(theta);
            let thetaGrad: Tensor = null;
            thetaT.backward = (g) => thetaGrad = g;

            const outGrad = Tensor.fromData([
                [1.0, -2.3, 3.1], [-0.53, 0.35, 0.837], [-0.7, 0.82, -0.9],
            ]);
            const objective = (x: Tensor) => {
                const mat = rotation(axis, x);
                return mat.mul(outGrad).sum();
            };

            objective(thetaT).backward(Tensor.fromData(1));

            const epsilon = 1e-2;
            const o1 = objective(Tensor.fromData(theta + epsilon)).data[0];
            const o2 = objective(Tensor.fromData(theta - epsilon)).data[0];
            const approxGrad = (o1 - o2) / (2 * epsilon);

            console.assert(
                // Even 1e-4 should work, but we don't want spurious failures.
                Math.abs(approxGrad - thetaGrad.data[0]) < 1e-2,
                approxGrad,
                thetaGrad.data[0],
            );
        });
    }
    console.log('[Done] rotation');
}

function testConv2d() {
    const input = Tensor.fromData([[[[0.12127403169870377, 1.5230209827423096, -1.4981558322906494, -1.274174690246582, 0.2041124701499939, -0.28717246651649475, 0.15605281293392181], [0.7922106981277466, -0.13447433710098267, -0.035242121666669846, 0.1419997215270996, 1.3321250677108765, -1.5926787853240967, 0.6496044397354126], [1.5718363523483276, 1.3930013179779053, -0.4733511507511139, -1.891462802886963, 0.059607818722724915, -0.1784893125295639, 0.9454407691955566], [0.49486836791038513, -0.16344070434570312, -0.2437339723110199, 0.7554162740707397, -0.3232586979866028, 0.16932861506938934, -1.499666452407837], [-0.2527104914188385, 0.06228289008140564, 1.2154278755187988, -0.5443894863128662, -0.0939042791724205, -0.14414571225643158, -0.7272892594337463]], [[2.547217607498169, 0.5766552090644836, -0.13145016133785248, 0.17669452726840973, 0.9758630990982056, -0.8622809052467346, -0.28951749205589294], [1.8719658851623535, -1.3668968677520752, -0.06438059359788895, 1.6987223625183105, 0.8319803476333618, -0.9437943696975708, 0.19236738979816437], [0.1381470412015915, -0.3365671932697296, 0.5937271118164062, 0.4362679123878479, -1.3113270998001099, -0.006806942168623209, 0.6468415856361389], [-0.5517890453338623, 0.8155103325843811, 0.8643335103988647, 0.17437538504600525, -0.385309100151062, -2.018650770187378, -0.13273927569389343], [0.05173168703913689, -1.1559079885482788, -0.3964315354824066, 0.9588875770568848, -1.4512158632278442, 0.009924361482262611, 0.1763177067041397]]], [[[-1.8798294067382812, 0.881496250629425, 1.303357481956482, 0.06611747294664383, 0.886265754699707, 0.5190923810005188, 0.1565951555967331], [0.16688783466815948, 0.9524247050285339, 0.9986836910247803, -1.5429205894470215, 0.9294336438179016, -0.039185069501399994, -0.45480236411094666], [-0.7435222864151001, -0.19538454711437225, 0.1281188428401947, -1.7863655090332031, 0.11827175319194794, -0.9757482409477234, -0.7389287948608398], [1.2820112705230713, 1.08162260055542, -0.8691518306732178, 0.6755786538124084, -1.2141934633255005, 0.4478440284729004, -1.7007497549057007], [0.6560084223747253, -0.6633696556091309, -3.4393808841705322, -0.24740757048130035, 1.7937321662902832, 1.7027539014816284, 0.6166509985923767]], [[-0.044284019619226456, -0.3738842308521271, -2.1908347606658936, -1.698687195777893, 0.5585806369781494, -0.5492671132087708, 0.5035452246665955], [-0.6287254691123962, 0.13515403866767883, -0.22161513566970825, 0.03274943307042122, -1.4317905902862549, 1.744746446609497, -0.28485774993896484], [1.1718746423721313, -0.30416756868362427, 0.2793470025062561, -1.656864881515503, 0.09120845049619675, -0.32787105441093445, -1.012520432472229], [0.45266297459602356, -0.9563449025154114, 0.25445136427879333, 0.12204378843307495, -0.18011587858200073, 1.9176952838897705, 1.2402234077453613], [-0.2513673007488251, -1.8624248504638672, 1.1320310831069946, 1.2630820274353027, -1.7574195861816406, -0.24128340184688568, 0.5600795149803162]]]]);
    const weight = Tensor.fromData([[[[0.07662881165742874, 0.10680202394723892, 0.1196611151099205], [0.12925124168395996, -0.06701194494962692, -0.16632303595542908], [-0.060186319053173065, -0.12194375693798065, -0.06083492934703827]], [[-0.13685239851474762, -0.08181580156087875, 0.12620003521442413], [0.11880825459957123, 0.03144823759794235, -0.03346917778253555], [0.054275672882795334, -0.10436287522315979, 0.20831206440925598]]], [[[-0.04318326711654663, -0.11970426887273788, -0.22976917028427124], [0.005024466197937727, -0.03034050576388836, 0.01687135547399521], [-0.144709512591362, 0.20082645118236542, -0.030369136482477188]], [[-0.023394450545310974, 0.1956133246421814, -0.18031714856624603], [0.035545360296964645, 0.16883423924446106, 0.01420146506279707], [-0.10752607882022858, 0.11332829296588898, -0.06178514286875725]]], [[[-0.221758171916008, -0.14777927100658417, -0.15731945633888245], [0.0074539510533213615, 0.1115729957818985, -0.20253951847553253], [0.0066865975968539715, -0.07964116334915161, -0.1156318187713623]], [[0.221797376871109, -0.13719797134399414, -0.1251373589038849], [-0.16215236485004425, -0.20889779925346375, 0.013641305267810822], [0.05452360957860947, -0.1655159592628479, -0.15547583997249603]]], [[[-0.08193732053041458, 0.21250039339065552, -0.13998755812644958], [0.139570415019989, -0.22510342299938202, -0.15314136445522308], [0.17542670667171478, -0.07151623070240021, -0.07793359458446503]], [[-0.12998360395431519, 0.17454133927822113, 0.19194470345973969], [0.10484809428453445, -0.2261330932378769, -0.14841625094413757], [-0.07537989318370819, 0.09379781037569046, 0.18162934482097626]]]]);
    const bias = Tensor.fromData([-0.12372291833162308, 0.004431994166225195, -0.1764194369316101, -0.0016671607736498117]);
    const expected = Tensor.fromData([[[[-0.8928871154785156, 0.734058678150177, -0.4391170144081116, -0.3375074565410614], [-0.5252106189727783, 0.7603395581245422, -1.175940752029419, -0.052886463701725006], [0.10452472418546677, -0.39694419503211975, -0.2993341386318207, 0.0530325211584568]], [[0.9244762063026428, 0.07965043187141418, 0.41107413172721863, 0.4029405415058136], [0.5389376282691956, -0.2699183523654938, 0.1605948507785797, 0.010881250724196434], [-0.2712033689022064, -0.15450073778629303, 0.04303787276148796, 0.24935926496982574]], [[-1.1404660940170288, -0.4802805185317993, -0.16857606172561646, -0.10653206706047058], [-0.5543047785758972, -0.5882198810577393, 0.8566713333129883, -0.1523425579071045], [-0.3175699710845947, 0.34713536500930786, 0.18380680680274963, -0.5425316691398621]], [[-1.0426572561264038, 1.1806315183639526, -0.4233122169971466, -0.33848387002944946], [-0.2248595654964447, 0.8119769096374512, 0.05570347607135773, 0.3150782287120819], [0.3938003182411194, -0.4228868782520294, -0.2388833612203598, 0.009767919778823853]]], [[[-0.1177658960223198, -0.2125471979379654, 0.06979022920131683, 0.06569939106702805], [-0.2626723051071167, 0.11318999528884888, 0.43099483847618103, -0.38306906819343567], [0.10570292174816132, 0.030578190460801125, -0.24050942063331604, -0.467252641916275]], [[-0.01145907212048769, -0.36894649267196655, 0.15029588341712952, -0.2377733290195465], [0.16605594754219055, -0.10379000753164291, -1.2110472917556763, -0.6730716824531555], [-0.236545592546463, 0.19638673961162567, -0.6483603715896606, 0.4622504413127899]], [[-0.6009157299995422, 0.6014948487281799, -0.14321595430374146, 0.00703446613624692], [-0.727310299873352, -0.008084127679467201, 0.2503087818622589, 0.5382304787635803], [-0.2438088059425354, -0.9122434854507446, -0.3930203914642334, 0.23440586030483246]], [[0.23141370713710785, 0.7170573472976685, -0.6482881903648376, -0.2685113847255707], [-0.5131137371063232, 0.9842191338539124, 0.6289372444152832, 0.02509705349802971], [0.30196627974510193, -0.09661834686994553, -0.1903565675020218, -0.4856777787208557]]]]);
    const result = conv2d(weight, bias, input, 2);
    console.assert(expected.shape.equals(result.shape), expected.shape, result.shape);
    console.assert(
        !result.data.some((x, i) => Math.abs(x - expected.data[i]) > 1e-5),
        result.data,
        expected.data,
    )

    console.log('[Done] conv2d');
}

function assertEqual(t1: Tensor, t2: Tensor) {
    console.assert(t1.shape.equals(t2.shape), t1.shape, t2.shape);
    const bad = t1.data.some((x, i) => x != t2.data[i]);
    console.assert(!bad, t1.data, t2.data);
}

function runTests() {
    testTranspose();
    testLinear();
    testSlice();
    testAccumGrad();
    testCat();
    testSinCos();
    testPow();
    testExp();
    testReLU();
    testRotation();
    testConv2d();
};
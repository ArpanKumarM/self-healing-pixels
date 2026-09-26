// Runs one NCA step with the browser code (docs/nca.js) under Node and writes the result.
// Driven by tests/test_nca.py, which compares it against PyTorch.
const fs = require("fs");
const tf = require("@tensorflow/tfjs");
const { createNCA } = require("../docs/nca.js");

const [inPath, outPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inPath, "utf8"));
const nca = createNCA(tf, input.model);
nca.setAngle(input.angle);
const x = tf.tensor4d(input.state, [1, input.size, input.size, input.model.channels]);
const y = nca.step(x, 1.0);  // fire rate 1: every cell updates, so the step is deterministic
fs.writeFileSync(outPath, JSON.stringify(Array.from(y.dataSync())));

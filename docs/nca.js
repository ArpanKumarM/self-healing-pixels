// Neural Cellular Automaton step in TensorFlow.js. Mirrors src/self_healing_pixels/nca.py exactly.
// Works in the browser (window.createNCA) and in Node (require) so it can be tested against PyTorch.
(function (root) {
  const ALIVE_THRESHOLD = 0.1;

  function perceptionFilter(tf, channels, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]].map(r => r.map(v => v / 8));
    const sy = [0, 1, 2].map(i => [0, 1, 2].map(j => sx[j][i]));  // transpose
    const id = [[0, 0, 0], [0, 1, 0], [0, 0, 0]];
    const kx = sx.map((r, i) => r.map((v, j) => c * v - s * sy[i][j]));
    const ky = sx.map((r, i) => r.map((v, j) => s * v + c * sy[i][j]));
    const kernels = [id, kx, ky];
    // Depthwise filter layout [3, 3, inChannels, multiplier]; output channel = ch * 3 + k.
    const data = new Float32Array(3 * 3 * channels * 3);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let ch = 0; ch < channels; ch++)
          for (let k = 0; k < 3; k++)
            data[((i * 3 + j) * channels + ch) * 3 + k] = kernels[k][i][j];
    return tf.tensor4d(data, [3, 3, channels, 3]);
  }

  function createNCA(tf, model) {
    const C = model.channels, H = model.hidden;
    const w1 = tf.tensor2d(model.w1, [3 * C, H]);
    const b1 = tf.tensor1d(model.b1);
    const w2 = tf.tensor2d(model.w2, [H, C]);
    let filter = perceptionFilter(tf, C, 0);

    function alive(x) {
      const alpha = x.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
      return tf.maxPool(alpha, 3, 1, "same").greater(ALIVE_THRESHOLD);
    }

    return {
      channels: C,
      setAngle(angle) {
        filter.dispose();
        filter = perceptionFilter(tf, C, angle);
      },
      // x: [1, size, size, C]. Returns the next state (caller disposes the old one).
      step(x, fireRate = 0.5, rand = null) {
        return tf.tidy(() => {
          const [, h, w] = x.shape;
          const preAlive = alive(x);
          const y = tf.depthwiseConv2d(x, filter, 1, "same").reshape([h * w, 3 * C]);
          const dx = tf.relu(y.matMul(w1).add(b1)).matMul(w2).reshape([1, h, w, C]);
          const r = rand || tf.randomUniform([1, h, w, 1]);
          const next = x.add(dx.mul(r.lessEqual(fireRate).cast("float32")));
          const life = tf.logicalAnd(preAlive, alive(next)).cast("float32");
          return next.mul(life);
        });
      },
      dispose() {
        [w1, b1, w2, filter].forEach(t => t.dispose());
      },
    };
  }

  function seedState(tf, size, channels) {
    const data = new Float32Array(size * size * channels);
    const centre = (Math.floor(size / 2) * size + Math.floor(size / 2)) * channels;
    for (let ch = 3; ch < channels; ch++) data[centre + ch] = 1;
    return tf.tensor4d(data, [1, size, size, channels]);
  }

  const api = { createNCA, seedState, perceptionFilter };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof self !== "undefined" ? self : this);

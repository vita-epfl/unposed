const params = new URLSearchParams(location.search);

const DIM = 2;
const N_JOINTS = 13;

if ((params.get('model') || '').toLowerCase() == 'large') {
    IN_N = 50;
    OUT_N = 5;
    MODEL = 'models/model.onnx';
    USE_CONDITIONAL_MASKING = false;
    document.querySelector('.model-name').innerText = 'default';
} else if ((params.get('model') || '').toLowerCase() == 'long') {
    IN_N = 55;
    OUT_N = 20;
    MODEL = 'models/model_full_l.onnx';
    USE_CONDITIONAL_MASKING = false;
    document.querySelector('.model-name').innerText = 'long';
} else {
    IN_N = 50;
    OUT_N = 5;
    MODEL = 'models/model_s_2d_small.onnx';
    USE_CONDITIONAL_MASKING = false;//true;
    document.querySelector('.model-name').innerText = 'fast';
}


loadModel = async () => ort.InferenceSession.create(MODEL, { executionProviders: ['wasm'], /* options: webnn, webgl, wasm, webgpu */ });

function preprocess(inputData) {
    inputData = inputData.flat(Infinity);


    var mask = [...new Array(IN_N * DIM * N_JOINTS).fill(1), ...new Array(OUT_N * DIM * N_JOINTS).fill(0)];
    if (USE_CONDITIONAL_MASKING) {
        for (let i = 0; i < inputData.length; i+=2) {
            if (inputData[i] != 0 && inputData[i+1] != 0) {
                mask[i] = 1;
                mask[i+1] = 1;
            }
        }
    }
    mask = new Float32Array(mask);

    const max = Math.max(...inputData);
    const min = Math.min(...inputData);
    inputData = inputData.map(value => (2 * (value - min) / (max - min)) - 1);

    var timepoints = new Float32Array(new Array(IN_N + OUT_N).fill(0).map((x, i) => i));
    const pose = new Float32Array([...inputData, ...new Array(DIM * N_JOINTS * OUT_N).fill(0)]);

    return [pose, mask, timepoints, min, max];
}

async function runModel(session, pose, mask, timepoints) {
    const poseTensor = new ort.Tensor('float32', pose, [1, IN_N + OUT_N, DIM * N_JOINTS]); // Adjust shape if necessary
    const maskTensor = new ort.Tensor('float32', mask, [1, IN_N + OUT_N, DIM * N_JOINTS]); // Adjust shape if necessary
    const timepointsTensor = new ort.Tensor('float32', timepoints, [1, IN_N + OUT_N]); // Adjust shape if necessary

    const feeds = { pose: poseTensor, mask: maskTensor, timepoints: timepointsTensor };
    const results = await session.run(feeds);

    return results.output.data;
}

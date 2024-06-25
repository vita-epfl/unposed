let video = document.querySelector('video');
const realTimeCanvas = document.getElementById('realTimeCanvas');
const poseCanvas = document.getElementById('poseCanvas');
const predCanvas = document.getElementById('predictionAnalysis');
const realTimeCtx = realTimeCanvas.getContext('2d');
const poseCtx = poseCanvas.getContext('2d');
const predCtx = predCanvas.getContext('2d');

predCanvas.style.width = '90vw';
predCanvas.style.height = '9vw';
predCanvas.style.border = 'none';
predCanvas.width = parseFloat(window.getComputedStyle(predCanvas).getPropertyValue('width'));
predCanvas.height = parseFloat(window.getComputedStyle(predCanvas).getPropertyValue('height'));

poseCanvas.width = parseFloat(window.getComputedStyle(poseCanvas).getPropertyValue('width'));
poseCanvas.height = parseFloat(window.getComputedStyle(poseCanvas).getPropertyValue('height'));

realTimeCanvas.width = poseCanvas.width;
realTimeCanvas.height = poseCanvas.height;

realTimeCtx.translate(realTimeCanvas.width, 0);
realTimeCtx.scale(-1, 1);

let BASE_URL = location.origin;
const LOW_CONFIDENCE_THRESHOLD = 0.1; // OpenPifPaf confidence lower bound for drawing joint (and counting as un-occluded if USE_CONDITIONAL_MASKING is true)
const OPP_WIDTH = 520; // lower resolution for openpifpaf model
const OPP_HEIGHT = 400;

let pendingPromises = [];
initializeWebSocket();

const frameHistory = [];
const imageHistory = [];
let inferenceLatencies = [];
let activePrediction = null;
let latestPrediction = null;
let analysisFrozen = false;

let isPredicting = false;

let numShifts = 0;
let modelSession;

drawSequence();
function drawSequence() {
    if (!activePrediction || !analysisFrozen) activePrediction = latestPrediction;
    if (!activePrediction) return window.requestAnimationFrame(drawSequence);

    predCtx.clearRect(0, 0, predCanvas.width, predCanvas.height);
    let fontSize = 16;

    let scale = (predCanvas.height - fontSize) / poseCanvas.height;

    let widthPerFrame = poseCanvas.width * scale;
    let framesPerWidth = Math.floor(predCanvas.width / widthPerFrame);
    let excessWidth = predCanvas.width - framesPerWidth * widthPerFrame;

    if (framesPerWidth < OUT_N + 1) return;

    predCtx.save();
    
    predCtx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    predCtx.filter = 'blur(50px)';
    predCtx.fillRect(0, 0, predCanvas.width, predCanvas.height - fontSize);

    predCtx.filter = 'none';
    predCtx.translate(excessWidth, 0);
    predCtx.translate(widthPerFrame * framesPerWidth, 0);


    for (let f = activePrediction.length - 1; f >= activePrediction.length - framesPerWidth; f--) {
        let isFuture = f > IN_N - 1;

        predCtx.translate(-widthPerFrame, 0);
        predCtx.translate(-excessWidth/framesPerWidth, 0);

        predCtx.lineWidth = 2;
        predCtx.globalAlpha = 0.2;
        drawGrid(predCtx, 0, widthPerFrame, 0, predCanvas.height - fontSize);
        predCtx.globalAlpha = 1;

        var keypoints = Array.from(activePrediction[f]);

        for (let i = 0; i < keypoints.length; i+=3) {
            keypoints[i] *= scale;
            keypoints[i+1] *= scale;
        }

        drawPose(predCtx, keypoints, isFuture, 1/3);
        
        let bottomPadding = 4;
        predCtx.font = `${fontSize - bottomPadding}px sans-serif`;
        predCtx.fillStyle = 'white';
        let text = `t = ${f - IN_N + 1} (${isFuture ? 'predicted' : 'observed'})`;
        let measure = predCtx.measureText(text);
        predCtx.clearRect(0, predCanvas.height - fontSize, widthPerFrame, fontSize);
        predCtx.fillText(text, (widthPerFrame - measure.width) / 2, predCanvas.height - bottomPadding);

    }
    predCtx.restore();

    window.requestAnimationFrame(drawSequence);
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.querySelector('video');
        video.srcObject = stream;
        // video.src = 'example1.mp4'

        video.play();

        video.addEventListener('loadeddata', () => {
            // setInterval(() => captureFrame(video), 1000 / 20);
            captureFrame(video);
        });
    } catch (e) {
        document.querySelector('.collecting-frames').innerText = 'Must allow camera access to use'
        document.querySelector('.collecting-frames').style.color = 'red';
    }
}

// todo: option to choose openpifpaf subject other than highest confidence
function chooseAnnotation(poseData, time, totalFrames) { return poseData; }

fpsFrames = 0;
totalFrames = 0;

async function captureFrame(video) {
    if (typeof fpsInterval === 'undefined' && frameHistory.length >= IN_N) {
        let int = 1_000;
        fpsFrames = 0;
        fpsInterval = setInterval(() => {
            document.querySelector('span.fps').innerText = (fpsFrames / (int / 1_000)).toFixed(0);
            document.querySelector('span.latency').innerText = (inferenceLatencies.reduce((a, b) => a + b, 0) / (inferenceLatencies.length || 1)).toFixed(0) + 'ms';
            fpsFrames = 0;
        }, int);
    }
    fpsFrames++;
    totalFrames++;

    realTimeCtx.drawImage(video, 0, 0, realTimeCanvas.width, realTimeCanvas.height);
    const frame = realTimeCanvas.toDataURL('image/jpeg');

    let _width = realTimeCanvas.width;
    let _height = realTimeCanvas.height;
    realTimeCanvas.width = OPP_WIDTH;
    realTimeCanvas.height = OPP_HEIGHT;
    realTimeCtx.translate(realTimeCanvas.width, 0);
    realTimeCtx.scale(-1, 1);
    let scaleX = _width / realTimeCanvas.width;
    let scaleY = _height / realTimeCanvas.height;
    realTimeCtx.drawImage(video, 0, 0, realTimeCanvas.width, realTimeCanvas.height);
    const lowResFrame = realTimeCanvas.toDataURL('image/jpeg');
    realTimeCanvas.width = _width;
    realTimeCanvas.height = _height;
    realTimeCtx.translate(realTimeCanvas.width, 0);
    realTimeCtx.scale(-1, 1);

    let poseData;
    try {
        let time = video.currentTime;
        poseData = await estimatePose(lowResFrame);
        poseData = chooseAnnotation(poseData, time, totalFrames);
        for (let i = 0; i < poseData.annotations[0].keypoints.length; i+=3) {
            poseData.annotations[0].keypoints[i] *= scaleX;
            poseData.annotations[0].keypoints[i+1] *= scaleY;
        }
    } catch (e) {
        return setTimeout(() => captureFrame(video), 100);
    }

    if (frameHistory.length < IN_N) {
        document.querySelector('span.collecting-frames').innerText = `Collecting frames ${frameHistory.length + 1}/${IN_N}`;
    } else {
        document.querySelector('span.collecting-frames').classList.add('hidden');
        poseCanvas.classList.remove('hidden');
        document.querySelector('#predictionAnalysis').classList.remove('hidden');
        document.querySelector('.btns').classList.remove('hidden');
    }
    frameHistory.push(poseData);
    imageHistory.push(frame);

    if (frameHistory.length > IN_N) {
        numShifts++;
        frameHistory.shift();
        imageHistory.shift();
    }

    if (frameHistory.length >= IN_N && !isPredicting) {
        numShifts = 0;
        isPredicting = true;
        await predictFuturePoses();
    }
    await captureFrame(video);
}

async function predictFuturePoses() {
    if (!modelSession) modelSession = await loadModel(); // load the model if not already loaded

    let hist = frameHistory.slice(-IN_N); // get the last in_n frames
    let [pose, mask, timepoints, min, max] = preprocess(hist.map(x => oppToDeposit(x))); // You'll need to implement preprocess based on model input expectations

    // calculate inference latency
    let t = Date.now();
    const outputData = await runModel(modelSession, pose, mask, timepoints);
    inferenceLatencies.push(Date.now() - t);
    if (inferenceLatencies.length > 5) inferenceLatencies.shift();

    let reshapedOutputData = new Array(DIM * N_JOINTS);
    for (let i = 0; i < DIM * N_JOINTS; i++) reshapedOutputData[i] = outputData.slice(i * (IN_N + OUT_N), (i + 1) * (IN_N + OUT_N));

    // denormalize the data back to the original scale
    denormalizedOutput = reshapedOutputData.map(row => row.map(value => (value + 1) / 2 * (max - min) + min));

    // transpose output
    predictions = Array.from({ length: denormalizedOutput[0].length }, () => new Float32Array(denormalizedOutput.length));
    for (let i = 0; i < denormalizedOutput.length; i++) {
        for (let j = 0; j < denormalizedOutput[i].length; j++) {
            predictions[j][i] = denormalizedOutput[i][j];
        }
    }

    latestPrediction = [...hist.map(x => x.annotations[0].keypoints), ...predictions.slice(IN_N).map(depositToOpp)];

    const image = new Image();
    image.onload = () => {
        isPredicting = false;

        poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height); // reset canvas
        poseCtx.drawImage(image, 0, 0, poseCanvas.width, poseCanvas.height); // draw corresponding input image
        drawPose(poseCtx, hist[hist.length-1].annotations[0].keypoints, false); // draw latest ground truth pose
        drawPose(poseCtx, depositToOpp(predictions[predictions.length-1]), true); // draw furthest out prediction pose

        let latestGroundTruth = predictions[IN_N - 1];
        let furthestPrediction = predictions[predictions.length - 1];

        // movement compass
        const computeDirectionShift = (prev, future) => {
            let dx = 0, dy = 0, n = 0;
            for (let i = 0; i < prev.length; i += 2) {
                if (future[i] < 0 || future[i+1] < 0 || future[i] > realTimeCanvas.width || future[i+1] > realTimeCanvas.height) continue;
                dx += future[i] - prev[i];
                dy += future[i+1] - prev[i+1];
                n++;
            }
            return [dx / n, dy / n];
        }

        let [dx, dy] = computeDirectionShift(latestGroundTruth, furthestPrediction);

        let maxMagnitude = 20;
        let padding = 10;
        dx = Math.min(Math.max(dx, -maxMagnitude), maxMagnitude);
        dy = Math.min(Math.max(dy, -maxMagnitude), maxMagnitude);
        drawArrow(poseCtx, maxMagnitude + padding, maxMagnitude + padding, dx, dy, maxMagnitude);
    };
    image.src = imageHistory[imageHistory.length - 1 - numShifts];
}


/***********************************/
/** OpenPifPaf <-> DePOSit Format */
/*********************************/

function oppToDeposit(frame) {
    frame = frame.annotations[0]?.keypoints;
    if (!frame) return new Array(26).fill(0);
    let out = [];

    for (let kp of [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]) {
        if (!USE_CONDITIONAL_MASKING) {
            out.push(...[frame[kp * 3], frame[kp * 3 + 1]]);
            continue;
        }

        if (frame[kp * 3 + 2] < LOW_CONFIDENCE_THRESHOLD) out.push(...[0, 0]);
        else out.push(...[frame[kp * 3], frame[kp * 3 + 1]]);
    }

    return out;
}
function depositToOpp(frame) {
    let out = [];

    for (let i = 0; i < 51; i++) out.push(0);

    for (let i = 0; i < 13; i++) {
        let kp = [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10][i];
        out[kp * 3] = frame[i * 2];
        out[kp * 3 + 1] = frame[i * 2 + 1];
        out[kp * 3 + 2] = 1
    }

    return out;
}

/*********************************/
/** OpenPifPaf/Pose Estimation **/
/*******************************/
// todo: use onnx model from https://openpifpaf.github.io/openpifpafwebdemo/ instead of streaming to/from websocket

async function estimatePose(frame) {
    return new Promise((resolve, reject) => {
        if (socket.readyState === WebSocket.OPEN) {
            pendingPromises.push({ resolve, reject });
            socket.send(frame);
        } else {
            reject(new Error('Socket is not open.'));
        }
    });
}

function initializeWebSocket() {
    socket = new WebSocket('wss://vitademo.epfl.ch/movements/pifpaf/pifpaf_accurate/ws');
    socket.addEventListener('open', (event) => {
        console.log('Connected to WebSocket');
    });
    socket.addEventListener('message', (event) => {
        if (pendingPromises.length > 0) {
            const { resolve } = pendingPromises.shift();
            resolve(JSON.parse(event.data));
        }
    });
    socket.addEventListener('close', (event) => {
        console.log('Disconnected from WebSocket');
    });
    socket.addEventListener('error', (event) => {
        console.error('WebSocket error observed:', event);
    });
}

initializeWebSocket();
startCamera();

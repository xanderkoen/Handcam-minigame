import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
const demosSection = document.getElementById("demos");
let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let detectButton;
let gameButton;
let correctpredictions = 0
let totaltestposes = 0
let accuracy;
let webcamRunning = false;
let gameIsRunning = false;
let handData = [];
let timer = null;
let currentPose = null;
let previousPose = null;
let detectedPose = null;
let gametime = 0;
let knownposes = ["thumb", "fist", "paper"];
import jsondata from './data.json' with { type: "json"};

let data = [] ;

//check voor foutieve data en schoon het evt op
for (let i = 0; i < jsondata.data.length; i++) {

    //check of de posedata uit 63 poses bestaat
    if (jsondata.data[i].pose.length === 63) {
        data.push(jsondata.data[i]);
    }else {
        //console log foute data
        console.log("Data niet toegevoegd " + jsondata.data[i].label + " object " + i + " in de json")
    }
}

//volledige posedata set
console.log(data)

//shuffle data
data = data.sort(() => Math.random() - 0.5);

//split data
const train = data.slice(0, Math.floor(data.length * 0.8));
const test = data.slice(Math.floor(data.length * 0.8) + 1);

import kNear from "./knear.js"

const k = 3
const machine = new kNear(k);

const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
}
//trainNN()
nn.load(modelDetails, () => console.log("het model is geladen!"))

function trainNN() {
    for (let i = 0; i < train.length; i++) {
        //schoon evt foutieve Data op
        if (train[i].pose.length === 63) {
            //voeg data toe aan neural network
            const {pose, label} = data[i];
            nn.addData(pose, {label});
        }
    }

    const trainingOptions = {
        detectionConfidence: 0.8,
        epochs: 250,
        learningRate: 0.3,
        hiddenUnits: 16,
    }

    nn.normalizeData()
    nn.train(trainingOptions, () => finishedTraining())
}

async function finishedTraining(){
    const results = await nn.classify([0.6415433883666992, 0.7114764451980591, -6.254308573261369e-7, 0.5688923597335815, 0.6896818280220032, 0.009708156809210777, 0.5334256887435913, 0.6452322602272034, 0.006482671480625868, 0.5055701732635498, 0.613501787185669, -0.00344936759211123, 0.47866472601890564, 0.5819242000579834, -0.009533806703984737, 0.5592749118804932, 0.5088087320327759, 0.02581973560154438, 0.4839344024658203, 0.5430740118026733, -0.002450814237818122, 0.5024495720863342, 0.5912519693374634, -0.023267876356840134, 0.5306638479232788, 0.6039349436759949, -0.03325340896844864, 0.566989541053772, 0.5039845108985901, 0.00935153104364872, 0.4784100651741028, 0.5496658086776733, -0.020906219258904457, 0.5050062537193298, 0.6048910021781921, -0.03101855330169201, 0.5358490347862244, 0.6129424571990967, -0.03066985122859478, 0.5717960000038147, 0.5138264298439026, -0.010310458950698376, 0.48743942379951477, 0.5641871094703674, -0.03420441225171089, 0.516165018081665, 0.6171844005584717, -0.026485437527298927, 0.5462178587913513, 0.6213974952697754, -0.015010000206530094, 0.5732951164245605, 0.5378371477127075, -0.03078080154955387, 0.507695198059082, 0.5825936794281006, -0.0444752499461174, 0.5288373827934265, 0.6165801286697388, -0.0388009287416935, 0.5559132099151611, 0.6174261569976807, -0.029290897771716118
    ]);
    console.log(`I think this is a ${results[0].label}`)

    testNN()

    //export model
    nn.save("model", () => console.log("model was saved!"))
}

async function testNN() {
    for (let i = 0; i < test.length; i++) {
        //voer de tests uit met de test data

        const prediction = await nn.classify(test[i].pose)

        if (prediction[0].label === test[i].label){
            console.log(`test voorspelling : ${prediction[0].label}. antwoord : ${test[i].label} | ✅`)
        }else{
            console.log(`test voorspelling : ${prediction[0].label}. antwoord : ${test[i].label} | ❌`)

        }

        //add stats for accuracy
        if (prediction[0].label === test[i].label) {
            totaltestposes++
            correctpredictions++
        }else{
            totaltestposes++
        }
    }

    accuracy = correctpredictions / totaltestposes

    console.log(`accuracy : ${accuracy * 100}%. totalposes : ${totaltestposes}. correct guesses : ${correctpredictions}`);
}

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const gamevideo = document.getElementById("gamewebcam");
const canvasElement = document.getElementById("output_canvas");
const gameCanvasElement = document.getElementById("output_game");
const canvasCtx = canvasElement.getContext("2d");
const gamecanvasCtx = canvasElement.getContext("2d");
// Check if webcam access is supported.
const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);

    detectButton = document.getElementById("testButton");
    detectButton.addEventListener("click", detectPose);

    gameButton = document.getElementById("gameButton");
    gameButton.addEventListener("click", startGame)
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}

async function detectPose () {
    let Detectedpose = await nn.classify(handData);
    console.log(`I think this is a ${Detectedpose[0].label}`)

    let answerText = document.getElementById("detectanswer");
    answerText.innerHTML = `I think this is a ${Detectedpose[0].label}`;
}
// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}


function flattenData(data) {
    return data.reduce((acc, obj) => {
        return acc.concat(Object.values(obj))
    }, []);
}

let lastVideoTime = -1;
let results = undefined;
console.log(video);
async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {

            //log landmarks
            handData = flattenData(landmarks)
            //console.log(handData)


            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#ee00ff",
                lineWidth: 5
            });
            drawLandmarks(canvasCtx, landmarks, { color: "#2b00ff", lineWidth: 2 });
        }
    }
    canvasCtx.restore();
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }

    if (timer != null) {
        await detectGame()
    }


}

function startGame() {
    let gametext = document.getElementById("gametext")
    let gamebutton = document.getElementById("gameButton")
    let gamescore = document.getElementById("gamescore")
    let gametimer = document.getElementById("gametimer")
    let gamediv = document.getElementById("gamediv");
    let answerText = document.getElementById("gamedetection");

    //remove invisibility from minigame
    gametext.style.visibility = "visible"
    gamediv.style.visibility = "visible"
    answerText.style.visibility = "visible"

    gameIsRunning = true;
    let score = 0;

    //controleer pose met antwoord
    let checkAnswer = setInterval( function () {
        if (gametime > 0) {
            if (checkPoseAnswer(currentPose)) {
                let addition = 100 / 100 * gametime
                score = score + 500 + addition; //standaard score van 500 + addition als time bonus
                gamescore.innerHTML = score; //update score text
                currentPose = getRandomPose()
                gametext.innerHTML = "show me a :" + currentPose
            }

        }
    }, 100)

    if (timer !== null) {
        clearInterval(timer); // Stop de timer als deze al loopt
        clearInterval(checkAnswer)
        console.log("Timer stopped and reset.");
        gamebutton.innerHTML = "START MINIGAME";
        gametimer.innerHTML = "MINIGAME STOPPED";
        gameIsRunning = false;
        score = 0;
        enableCam()
        timer = null; //reset timer
        return;
    }else{
        gamebutton.innerHTML = "STOP MINIGAME";
        enableCam()
    }

    gametime = 30;
    timer = setInterval(function () {
        gametime--;
        if (gametime < 0) {
            clearInterval(timer);
            clearInterval(checkAnswer)
            console.log("countdown finished");
            gametimer.innerHTML = "GAME OVER";
            enableCam() // automatically disable predictions on game end to avoid no predictions on replay
            gamebutton.innerHTML = "START MINIGAME";
            gameIsRunning = false;
            score = 0;
            timer = null; //reset de timer
        } else {
            gametimer.innerHTML = "Time : " + gametime;
        }
    }, 1000);

    //get pose the player needs to do
    currentPose = getRandomPose()
    gametext.innerHTML = "show me a :" + currentPose

}

async function detectGame () {
    let Detectedpose = await nn.classify(handData);
    console.log(`I think this is a ${Detectedpose[0].label}`)

    detectedPose = Detectedpose[0].label

    let answerText = document.getElementById("gamedetection");
    answerText.innerHTML = `I think this is a ${Detectedpose[0].label}`;


}

function getRandomPose() {
    let randomIndex;
    do {
        // Genereer een willekeurig indexnummer tussen 0 en de lengte van het array
        randomIndex = Math.floor(Math.random() * knownposes.length);
    } while (knownposes[randomIndex] === previousPose); // Blijf doorgaan totdat een nieuwe pose wordt gekozen

    // Bewaar de huidige pose als de vorige pose voor de volgende keer
    previousPose = knownposes[randomIndex];

    // Geef het item op de willekeurig gekozen indexnummer terug
    return knownposes[randomIndex];
}

function checkPoseAnswer() {
    if (gameIsRunning === true) {
        if (currentPose === detectedPose) {
            console.log("antwoord goed");
            return true;
        }else {
            console.log("antwoord fout");
            return false;
        }
    }else {
        return false; //game is niet active
    }

}
const webcamElement = document.getElementById('webcam');
if (navigator.mediaDevices.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            webcamElement.srcObject = stream;
        })
        .catch(function (error) {
            console.log("Something went wrong!");
            console.log(error)
        });
}

// async function app() {
//     console.log('Loading mobilenet..');
//     //
//     // // Load the model.
//     // net = await mobilenet.load();
//     console.log('Successfully loaded model');
//
//     // Create an object from Tensorflow.js data API which could capture image
//     // from the web camera as Tensor.
//     const webcam = await tf.data.webcam(webcamElement);
//     while (true) {
//         const img = await webcam.capture();
//         // const result = await net.classify(img);
//         //
//         // document.getElementById('console').innerText = 'prediction: ${result[0].className}\nprobability: ${result[0].probability}';
//         // Dispose the tensor to release the memory.
//         // img.dispose();
//
//         // Give some breathing room by waiting for the next animation frame to
//         // fire.
//         await tf.nextFrame();
//     }
// }
//
// app()
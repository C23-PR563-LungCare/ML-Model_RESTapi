const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const app = express();
const port = 8000;

const upload = multer();

//const modelPath = 'https://storage.googleapis.com/bucket-for-ml-model/model.json';

app.post('/testingModel', upload.single('image'), async (req, res)=>{
    try {
        const model = await tf.loadLayersModel("https://storage.googleapis.com/bucket-for-ml-model/model.json");

         const imageBuffer = req.file.buffer;
         console.log(imageBuffer);
         const image = tf.node.decodeImage(imageBuffer);
         console.log(image);
         const processedImage= preprocessImage(image);
         const predictions = await model.predict(processedImage).array();

         res.json({predictions});
    } catch (error) {
        console.log(error);
    res.status(500).json({ error: 'An error occurred' });
        
    }
    console.log('predictions should be working');
})

function preprocessImage(image) {
    let rgbImage;    
  // Resize the image to a fixed size (e.g., 224x224)
  const resizedImage = tf.image.resizeBilinear(image, [ 150, 150 ]);
  // Normalize the pixel values to the range of [0, 1]
  //console.log(resizedImage.shape[2]);
  if(resizedImage.shape[2] !== 3){
     rgbImage = tf.image.grayscaleToRGB(resizedImage);
    }else{
         rgbImage = resizedImage;
    }
  const normalizedImage = rgbImage.div(255.0);
  //console.log(normalizedImage);
  // Add a batch dimension to the image
  const batchedImage = normalizedImage.expandDims(0);
  //console.log(batchedImage);

  
  return batchedImage;
}


app.get('/', (req,res) =>{
    res.status(200).send('berhasil');
})


app.listen(port, () =>{
    console.log("App Is Working");
})
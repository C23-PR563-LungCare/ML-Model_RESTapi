const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const app = express();
const PORT = process.env.PORT || 8000;

const upload = multer();

app.post('/testingModel', upload.single('image'), async (req, res)=>{
    try {
        const model = await tf.loadLayersModel(process.env.Link_Bucket);

         const imageBuffer = req.file.buffer;
         const image = tf.node.decodeImage(imageBuffer);
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

  const resizedImage = tf.image.resizeBilinear(image, [ 150, 150 ]);

  if(resizedImage.shape[2] !== 3){
     rgbImage = tf.image.grayscaleToRGB(resizedImage);
    }else{
         rgbImage = resizedImage;
    }
  const normalizedImage = rgbImage.div(255.0);
  const batchedImage = normalizedImage.expandDims(0);
  
  return batchedImage;
}


app.get('/', (req,res) =>{
    res.status(200).send('berhasil');
})


app.listen(PORT, () =>{
    console.log(`App is working and listenign to port ${PORT}`);
})
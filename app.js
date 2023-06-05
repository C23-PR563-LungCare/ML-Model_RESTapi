const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const multer = require('multer');
const app = express();
const PORT = process.env.PORT || 8000;

const upload = multer();

app.use(express.json({limit: "20MB"}));
app.use(express.urlencoded({extended: true}));

app.post('/testingModel', async (req, res)=>{
    try {
         console.log("berhasil");
        const model = await tf.loadLayersModel(process.env.Link_Bucket);
        const reqImage = req.body.image;
        const decReqImage = Buffer.from(reqImage, 'base64');
         
         const image = tf.node.decodeImage(decReqImage);
         const processedImage= preprocessImage(image);
         const predictions = await model.predict(processedImage).array();
         res.status(200).send({predictions});
    } catch (error) {
        //console.log(error);
        res.status(500).send({ error: 'An error occurred' });
        
    }
    //console.log('predictions should be working');
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
    console.log(`App is working and listening to port ${PORT}`);
})
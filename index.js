
const classifier = knnClassifier.create()
const webcamElement = document.getElementById("webcam")


let net
async function app(){
	
	console.log("Loading model...")
	net = await mobilenet.load()
	console.log("loaded model!!")
	
	const webcam = await tf.data.webcam(webcamElement)
	
	
	
	
	
	const addExample = async (classId) => {
		const img = await webcam.capture()
		
		const activation = net.infer(img,true)
		
		classifier.addExample(activation,classId)
		
		img.dispose()
	}
	
	
	document.getElementById("image1").addEventListener("click",()=> addExample(0))
	document.getElementById("image2").addEventListener("click",()=> addExample(1))
	document.getElementById("image3").addEventListener("click",()=> addExample(2))
	
	while(true)
	{
		if(classifier.getNumClasses() > 0)
		{
			const img = await webcam.capture()
			
			const activation = net.infer(img,"conv_preds")
			
			const result = await classifier.predictClass(activation)
			
			const classes = ["Image 1","Image 2","Image 3"]
			
			document.getElementById("console").innerText= 
			`
			prediction: ${classes[result.label]}\n
			probability: ${result.confidences[result.label]}
			`
			
			img.dispose()
		}
		
		await tf.nextFrame()
	}
}

app()
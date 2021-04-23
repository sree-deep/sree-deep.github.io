
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
	
	
	
    function init() {
               imgObj = document.getElementById('myImage');
               imgObj.style.position= 'relative'; 
               imgObj.style.left = '0px'; 
            }
            function moveRight() {
               imgObj.style.left = parseInt(imgObj.style.left) + 5 + 'px';
            }
			
			function moveLeft() {
               imgObj.style.left = parseInt(imgObj.style.left) - 5 + 'px';
            }
	init() 

		
	async function startmoving()
	{
		while(true)
	{
		if(classifier.getNumClasses() > 0)
		{
			const img = await webcam.capture()
			
			const activation = net.infer(img,"conv_preds")
			
			const result = await classifier.predictClass(activation)
			
			const classes = ["Right","left","Stop"]
			
			document.getElementById("console").innerText= 
			`
			prediction: ${classes[result.label]}\n
			probability: ${result.confidences[result.label]}
			`
			if(result.label==0)
			{
				moveRight()
			}
			else if(result.label==1)
			{
				moveLeft()
			}
			else if(result.label==2)
			{
				init()
			}
			img.dispose()
		}
		
		await tf.nextFrame()
	}
	}
	document.getElementById("start").addEventListener("click",()=> startmoving())
	
}

	
app()


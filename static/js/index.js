let image = document.getElementById('output');

let showPreview=function(event){
		image.src = URL.createObjectURL(event.target.files[0]);
		image.style.display="block";
}
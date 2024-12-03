// Función para manejar la respuesta del backend
async function handleResponse() {
    // Asegúrate de que el input tiene un archivo seleccionado
    const fileInput = document.getElementById("file-upload");
    if (!fileInput || fileInput.files.length === 0) {
        alert("Por favor, selecciona una imagen antes de enviar.");
        return;
    }

    const formData = new FormData();
    const file = fileInput.files[0];
    formData.append("file", file);
    const allowedTypes = ["image/jpeg", "image/png", "image/tiff"];
    const errorMessage = document.getElementById("error-message");

    if (file) {
        // Verificamos si el tipo de archivo es válido
        if (!allowedTypes.includes(file.type)) {
            errorMessage.style.display = "block";
            clearForm();
            return;
        } else {
            errorMessage.style.display = "none";
        }
    }
    console.log('Enviando imagen...');

    try {
        // Realizar la solicitud al backend
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData,
        });

        if (response.status !== 200) {
            throw new Error(`Error al procesar la imagen. Código: ${response.status}`);
        }

        const data = await response.json();
        console.log('Datos recibidos:', data);

        showModal(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Hubo un problema al procesar la imagen. Inténtalo de nuevo.');
    }
}
  
// Mostrar modal según la respuesta
function showModal(data) {
    const modal = document.getElementById('resultModal');
    const overlay = document.getElementById('modalOverlay');
    const resultIcon = document.getElementById('resultIcon');
    const resultMessage = document.getElementById('resultMessage');
    const resultProbability = document.getElementById('resultProbability');
    const resultClass = document.getElementById('resultClass');
    const imgNoDetected = '../img/no_tumor_detected.png';
    const imgDetected = '../img/tumor_detected.png';
    
    resultMessage.style.size = '20px';
    resultProbability.textContent = '';
    resultClass.textContent = '';

    if (data.tumor_class === 'no_tumor') {
        resultIcon.src = imgNoDetected;
        resultMessage.textContent = `No se detectó un tumor.`;
        resultProbability.textContent = `% DE FIABILIDAD: ${data.probabilidad.toFixed(2)}%`;
        resultProbability.style.color = 'green';
    } else if (data.probabilidad > 70){        
        resultIcon.src = imgDetected;
        resultMessage.textContent = `Se detectó un tumor. Características:`;
        resultProbability.textContent = `% DE FIABILIDAD: ${data.probabilidad.toFixed(2)}%`;
        resultProbability.style.color = 'red';
        resultClass.textContent = `Clase: ${data.tumor_class || 'NO CLASIFICADA'}`;
        resultClass.style.color = 'gray';
    }
    else {
        resultIcon.src = imgNoDetected;
        resultMessage.textContent = `No se detectó un tumor con certeza. Características:`;
        resultProbability.textContent = `% DE FIABILIDAD: ${data.probabilidad.toFixed(2)}%`;
        resultProbability.style.color = 'orange';
        resultClass.textContent = `Posible clase: ${data.tumor_class || 'NO CLASIFICADA'}`;
        resultClass.style.color = 'gray';
    }
    clearForm();
    overlay.classList.remove('hidden');
    overlay.style.display = 'block';
    modal.classList.remove('hidden');
}

function clearForm() {
    document.getElementById('file-image').classList.add("hidden");
    document.getElementById('notimage').classList.remove("hidden");
    document.getElementById('start').classList.remove("hidden");
    document.getElementById('response').classList.add("hidden");
    document.getElementById("uploadForm").reset();
}

function closeModal() {
    const modal = document.getElementById('resultModal');
    const overlay = document.getElementById('modalOverlay');
    overlay.classList.add('hidden');
    overlay.style.display = 'none';
    modal.classList.add('hidden');
}
  
// se adjuntar evento al formulario
document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    await handleResponse();
});
  
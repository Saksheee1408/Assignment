document.addEventListener("DOMContentLoaded", () => {
    const dropZoneElement = document.getElementById("drop-zone");
    const inputElement = document.getElementById("file-input");
    const previewContainer = document.querySelector(".preview-container");
    const imagePreview = document.getElementById("image-preview");
    const predictBtn = document.getElementById("predict-btn");
    const loadingDiv = document.getElementById("loading");
    const resultsSection = document.getElementById("results-section");
    const predictedClassSpan = document.getElementById("predicted-class");
    const exportBtn = document.getElementById("export-btn");
    const resetBtn = document.getElementById("reset-btn");
    
    let chartInstance = null;
    let currentFile = null;
    let predictionHistory = [];

    // Drag and Drop Logic
    dropZoneElement.addEventListener("click", () => {
        inputElement.click();
    });

    inputElement.addEventListener("change", (e) => {
        if (inputElement.files.length) {
            updateThumbnail(inputElement.files[0]);
        }
    });

    dropZoneElement.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZoneElement.classList.add("drop-zone--over");
    });

    ["dragleave", "dragend"].forEach((type) => {
        dropZoneElement.addEventListener(type, (e) => {
            dropZoneElement.classList.remove("drop-zone--over");
        });
    });

    dropZoneElement.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            updateThumbnail(e.dataTransfer.files[0]);
        }
        dropZoneElement.classList.remove("drop-zone--over");
    });

    function updateThumbnail(file) {
        // Validate
        if (!['image/jpeg', 'image/png'].includes(file.type)) {
            alert('Invalid file format. Only JPG and PNG are allowed.');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            alert('File is too large. Maximum size is 5MB.');
            return;
        }
        
        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            imagePreview.src = reader.result;
            imagePreview.classList.remove("hidden");
            dropZoneElement.classList.add("hidden");
            predictBtn.classList.remove("hidden");
            resultsSection.classList.add("hidden");
        };
    }

    // Prediction Logic
    predictBtn.addEventListener("click", async () => {
        if (!currentFile) return;
        
        predictBtn.classList.add("hidden");
        loadingDiv.classList.remove("hidden");
        
        const formData = new FormData();
        formData.append("file", currentFile);
        
        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            
            const result = await response.json();
            loadingDiv.classList.add("hidden");
            
            if (response.ok) {
                showResults(result);
                // Save to history
                predictionHistory.push({
                    filename: currentFile.name,
                    timestamp: new Date().toISOString(),
                    prediction: result.class,
                    confidence: (result.confidence * 100).toFixed(2) + '%'
                });
            } else {
                alert(result.error || "An error occurred during prediction.");
                predictBtn.classList.remove("hidden");
            }
        } catch (error) {
            console.error(error);
            alert("Network error.");
            loadingDiv.classList.add("hidden");
            predictBtn.classList.remove("hidden");
        }
    });
    
    function showResults(data) {
        resultsSection.classList.remove("hidden");
        predictedClassSpan.textContent = data.class;
        
        // Render Chart
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        if (chartInstance) {
            chartInstance.destroy();
        }
        
        const labels = Object.keys(data.all_confidences);
        const dataValues = Object.values(data.all_confidences).map(v => (v * 100).toFixed(1));
        
        // Highlight colors for max
        const bgColors = labels.map(label => label === data.class ? 'rgba(59, 130, 246, 0.8)' : 'rgba(255, 255, 255, 0.2)');
        
        Chart.defaults.color = '#f8fafc';
        Chart.defaults.font.family = 'Inter';
        
        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: dataValues,
                    backgroundColor: bgColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // Reset Logic
    resetBtn.addEventListener("click", () => {
        currentFile = null;
        inputElement.value = "";
        imagePreview.classList.add("hidden");
        imagePreview.src = "";
        resultsSection.classList.add("hidden");
        dropZoneElement.classList.remove("hidden");
    });

    // Export Logic (Bonus)
    exportBtn.addEventListener("click", () => {
        if (predictionHistory.length === 0) {
            alert("No history to export.");
            return;
        }
        
        const headers = ["Filename", "Timestamp", "Prediction", "Confidence"];
        const rows = predictionHistory.map(entry => [
            entry.filename,
            entry.timestamp,
            entry.prediction,
            entry.confidence
        ]);
        
        let csvContent = "data:text/csv;charset=utf-8," 
            + headers.join(",") + "\n"
            + rows.map(e => e.join(",")).join("\n");
            
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "prediction_history.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});

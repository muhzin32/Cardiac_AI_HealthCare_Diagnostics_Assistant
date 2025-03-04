const apiUrl = "http://127.0.0.1:5000/analyze";

document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const loading = document.getElementById("loading");
    const resultContainer = document.getElementById("resultContainer");
    const resultTable = document.getElementById("resultTable");

    if (!file) {
        alert("Please select a .mat file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    loading.classList.remove("d-none");
    resultContainer.classList.add("d-none");
    resultTable.innerHTML = ""; // Clear previous results

    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        loading.classList.add("d-none");

        if (result.success) {
            resultContainer.classList.remove("d-none");

            // Populate result table
            result.segment_labels.forEach((label, index) => {
                const row = `<tr><td>${index + 1}</td><td>${label}</td></tr>`;
                resultTable.innerHTML += row;
            });

            // Plot ECG signal
            plotECGSignal(result.time, result.signal, result.r_peaks);
        } else {
            alert(result.error);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Error processing file.");
        loading.classList.add("d-none");
    }
});

function plotECGSignal(time, signal, rPeaks) {
    const ctx = document.getElementById("ecgChart").getContext("2d");

    // Clear previous chart if exists
    if (window.ecgChartInstance) {
        window.ecgChartInstance.destroy();
    }

    window.ecgChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: time,
            datasets: [
                {
                    label: "ECG Signal",
                    data: signal,
                    borderColor: "blue",
                    borderWidth: 1,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: "R-peaks",
                    data: rPeaks.map(i => ({ x: time[i], y: signal[i] })),
                    backgroundColor: "red",
                    pointRadius: 4,
                    type: "scatter"
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: "Time (s)" } },
                y: { title: { display: true, text: "Amplitude" } }
            }
        }
    });
}

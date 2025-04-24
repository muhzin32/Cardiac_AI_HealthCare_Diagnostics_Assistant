const apiUrl = "http://127.0.0.1:5000/analyze";
const apiUrlAllLeads = "http://127.0.0.1:5000/analyze_all_leads";
const apiUrlManualLeads = "http://127.0.0.1:5000/analyze_manual_leads";

document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const files = fileInput.files;
    const loading = document.getElementById("loading");
    const resultContainer = document.getElementById("resultContainer");
    const resultTable = document.getElementById("resultTable");
    const ecgImage = document.getElementById("ecgImage");
    const generateReportBtn = document.getElementById("generateReportBtn");
const analysisTypeInput = "manual"; // Set analysis type to manual


    if (!files || files.length === 0) {
        alert("Please select the required files: either a .mat file, or for WFDB records, .dat and .hea files (the .atr file is optional).");
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }
    
    // Add selected leads to formData
    const selectedLeadsInput = document.getElementById("selectedLeadsInput");
    if (selectedLeadsInput.value) {
        formData.append("selected_leads", selectedLeadsInput.value);
    }
    
    // Add analysis type to formData and handle checkbox for all leads
const analysisType = analysisTypeInput; // This will always be "manual"

    formData.append("analysis_type", analysisType);
    
    // For file-based analysis, we need to save the file path
    formData.append("record_path", files[0].name);

    // Reset UI elements
    loading.classList.remove("d-none");
    resultContainer.classList.add("d-none");
    resultTable.innerHTML = "";
    ecgImage.src = "";
    generateReportBtn.classList.add("d-none");

    try {
        let response;
        let apiEndpoint = apiUrl;
        
        // Choose the appropriate API endpoint based on analysis type
        if (analysisType === "all") {
            apiEndpoint = apiUrlAllLeads;
        } else if (analysisType === "manual") {
            apiEndpoint = apiUrlManualLeads;
        }
        
        response = await fetch(apiEndpoint, {
            method: "POST",
            body: formData
        });
        
        const result = await response.json();
        console.log("API Response:", result);
        loading.classList.add("d-none");

        if (result.success) {
            resultContainer.classList.remove("d-none");

            // Handle different response formats based on analysis type
            if (analysisType === "all" || analysisType === "manual") {
                // Display final disease name only
                document.getElementById("finalDisease").innerText = result.final_disease;
                
                // Hide the final disease row if the disease is "Unclassified"
                const finalDiseaseRow = document.getElementById("finalDiseaseRow");
                if (result.final_disease === "Unclassified") {
                    finalDiseaseRow.classList.add("d-none");
                } else {
                    finalDiseaseRow.classList.remove("d-none");
                }
                
                // For these types, we might not have segment-level predictions
                resultTable.innerHTML = `<tr>
                                          <td>All Segments</td>
                                          <td>${result.final_disease}</td>
                                          <td>${result.confidence}%</td>
                                        </tr>`;
            } else {
                // Populate table with each segment's prediction and confidence
                if (result.labels && result.confidences) {
                    result.labels.forEach((label, index) => {
                        const row = `<tr>
                                        <td>Segment ${index + 1}</td>
                                        <td>${label}</td>
                                        <td>${result.confidences[index]}%</td>
                                     </tr>`;
                        resultTable.innerHTML += row;
                    });
                }
                // Display final disease name only
                document.getElementById("finalDisease").innerText = result.final_disease;
                
                // Hide the final disease row if the disease is "Unclassified"
                const finalDiseaseRow = document.getElementById("finalDiseaseRow");
                if (result.final_disease === "Unclassified") {
                    finalDiseaseRow.classList.add("d-none");
                } else {
                    finalDiseaseRow.classList.remove("d-none");
                }
            }
            
            // Update ECG plot image (append timestamp to avoid caching)
            ecgImage.src = result.plot_url + "?t=" + new Date().getTime();

            // Show "Generate Report" button
            generateReportBtn.classList.remove("d-none");
        } else {
            alert(result.error || "An error occurred during analysis.");
        }
    } catch (error) {
        console.error("Error:", error);
        loading.classList.add("d-none");
        
        // Show a more detailed error message
        if (error instanceof SyntaxError && error.message.includes("JSON")) {
            alert("Server returned invalid JSON. The response might be malformed.");
        } else if (error instanceof TypeError && error.message.includes("fetch")) {
            alert("Network error: Could not connect to the server.");
        } else {
            alert("Error processing file: " + error.message);
        }
    }
});

document.getElementById("generateReportBtn").addEventListener("click", async function() {
    try {
        const response = await fetch("/generate_report", { method: "POST" });
        const result = await response.json();

        if (result.success) {
            // Ensure fresh load by appending a timestamp
            window.open(result.report_url + "?t=" + new Date().getTime(), "_blank");
        } else {
            alert("Error generating report: " + result.error);
        }
    } catch (error) {
        console.error("Report Generation Error:", error);
        alert("Failed to generate the report.");
    }
});

// Update UI based on analysis type selection
document.getElementById("analysisType").addEventListener("change", function() {
    const analysisType = this.value;
    const numLeadsSelect = document.getElementById("numLeadsSelect").parentElement;
    const selectAllCheckbox = document.getElementById("selectAllCheckbox").parentElement;
    const manualLeadsBox = document.getElementById("manualLeadsBox");
    
    // Hide/show elements based on analysis type
    if (analysisType === "all") {
        numLeadsSelect.style.display = "none";
        selectAllCheckbox.style.display = "none";
        manualLeadsBox.style.display = "none";
        manualLeadsBox.innerHTML = "<p class='mb-0'>All 12 leads will be analyzed.</p>";
    } else if (analysisType === "manual") {
        numLeadsSelect.style.display = "block";
        selectAllCheckbox.style.display = "block";
        manualLeadsBox.style.display = "block";
        updateManualDropdowns(); // Refresh the dropdowns
    } else {
        numLeadsSelect.style.display = "none";
        selectAllCheckbox.style.display = "none";
        manualLeadsBox.style.display = "none";
        manualLeadsBox.innerHTML = "<p class='mb-0'>Standard analysis will be performed.</p>";
    }
});

// Function to update dropdowns for the selected default combination
function updateDropdownsForDefaultCombination(diseaseKey) {
  manualLeadsBox.innerHTML = ""; // Clear existing dropdowns
  const leads = defaultCombinations[diseaseKey];

  if (leads) {
    leads.forEach((lead) => {
      const dropdown = createDropdown("leadDropdown" + lead);
      dropdown.value = leadOptions.find(option => option.label.startsWith(lead)).value;
      manualLeadsBox.appendChild(dropdown);
    });
  } else {
    manualLeadsBox.innerHTML = "<p class='text-danger'>Invalid disease key or no leads available.</p>";
  }
}

// Event listener for the default combination dropdown
const defaultCombinationSelect = document.getElementById("defaultCombinationSelect");
defaultCombinationSelect.addEventListener("change", function () {
  const selectedDisease = this.value;
  if (selectedDisease) {
    updateDropdownsForDefaultCombination(selectedDisease);
  } else {
    manualLeadsBox.innerHTML = "<p class='text-muted'>Please select a disease key to view leads.</p>";
  }
});

// Show the default combination section when "Default Combinations" is selected
numLeadsSelect.addEventListener("change", function () {
  if (numLeadsSelect.value === "5") {
    document.getElementById("defaultCombinationSection").classList.remove("d-none");
    manualLeadsBox.innerHTML = "<p class='text-muted'>Select a disease key to populate leads.</p>";
  } else {
    document.getElementById("defaultCombinationSection").classList.add("d-none");
    updateManualDropdowns();
  }
});

// Function to create a dropdown for lead selection
function createDropdown(id) {
  const select = document.createElement("select");
  select.className = "form-select mb-2";
  select.id = id;
  leadOptions.forEach(option => {
    const opt = document.createElement("option");
    opt.value = option.value;
    opt.textContent = option.label;
    select.appendChild(opt);
  });
  return select;
}

// Initialize UI based on default analysis type
document.getElementById("analysisType").dispatchEvent(new Event("change"));

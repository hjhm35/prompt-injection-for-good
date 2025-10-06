// Document Generator JavaScript
let promptCounter = 0;
let bodyCounter = 0;
let variantCounter = 0;
let currentEditingPrompt = null;
let currentEditingBody = null;

// Data storage
let prompts = [];
let bodies = [];
let selectedFormats = [];

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    initializeFormatSelector();
    updateCombinationCount();
    
    // Set up content type radio buttons
    const textRadio = document.getElementById('textContent');
    const fileRadio = document.getElementById('fileContent');
    
    textRadio.addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('textContentSection').style.display = 'block';
            document.getElementById('fileContentSection').style.display = 'none';
        }
    });
    
    fileRadio.addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('textContentSection').style.display = 'none';
            document.getElementById('fileContentSection').style.display = 'block';
        }
    });
});

// Format selector functions
function initializeFormatSelector() {
    const formatCards = document.querySelectorAll('.format-card');
    formatCards.forEach(card => {
        card.addEventListener('click', function() {
            const format = this.getAttribute('data-format');
            toggleFormat(format);
        });
    });
}

function toggleFormat(format) {
    const card = document.querySelector(`[data-format="${format}"]`);
    
    if (selectedFormats.includes(format)) {
        selectedFormats = selectedFormats.filter(f => f !== format);
        card.classList.remove('selected');
    } else {
        selectedFormats.push(format);
        card.classList.add('selected');
    }
    
    updateCombinationCount();
}

// Prompt functions
function addPrompt() {
    currentEditingPrompt = null;
    document.getElementById('promptName').value = '';
    document.getElementById('promptText').value = '';
    document.getElementById('promptDescription').value = '';
    
    // Reset variants to default
    resetVariants();
    
    const modal = new bootstrap.Modal(document.getElementById('promptModal'));
    modal.show();
}

function editPrompt(id) {
    const prompt = prompts.find(p => p.id === id);
    if (!prompt) return;
    
    currentEditingPrompt = id;
    document.getElementById('promptName').value = prompt.name;
    document.getElementById('promptText').value = prompt.text;
    document.getElementById('promptDescription').value = prompt.description;
    
    // Load variants
    loadVariants(prompt.variants);
    
    const modal = new bootstrap.Modal(document.getElementById('promptModal'));
    modal.show();
}

function savePrompt() {
    const name = document.getElementById('promptName').value.trim();
    const text = document.getElementById('promptText').value.trim();
    const description = document.getElementById('promptDescription').value.trim();
    
    if (!name || !text) {
        alert('Please enter both prompt name and text');
        return;
    }
    
    // Collect variants
    const variants = [];
    const variantElements = document.querySelectorAll('#variantsContainer .style-editor');
    
    variantElements.forEach(element => {
        const variantName = element.querySelector('[data-field="name"]').value;
        const font = element.querySelector('[data-field="font"]').value;
        const size = parseInt(element.querySelector('[data-field="size"]').value);
        const style = element.querySelector('[data-field="style"]').value;
        const steganographic = element.querySelector('[data-field="steganographic"]').checked;
        
        variants.push({
            name: variantName,
            font: font,
            size: size,
            style: style,
            steganographic: steganographic
        });
    });
    
    const promptData = {
        id: currentEditingPrompt || ++promptCounter,
        name: name,
        text: text,
        description: description,
        variants: variants
    };
    
    if (currentEditingPrompt) {
        const index = prompts.findIndex(p => p.id === currentEditingPrompt);
        prompts[index] = promptData;
    } else {
        prompts.push(promptData);
    }
    
    renderPrompts();
    updateCombinationCount();
    
    const modal = bootstrap.Modal.getInstance(document.getElementById('promptModal'));
    modal.hide();
}

function deletePrompt(id) {
    if (confirm('Are you sure you want to delete this prompt?')) {
        prompts = prompts.filter(p => p.id !== id);
        renderPrompts();
        updateCombinationCount();
    }
}

function renderPrompts() {
    const container = document.getElementById('promptsContainer');
    const addButton = container.querySelector('.add-item-btn');
    
    // Clear existing prompt cards
    container.querySelectorAll('.item-card').forEach(card => card.remove());
    
    prompts.forEach(prompt => {
        const card = document.createElement('div');
        card.className = 'item-card';
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <h6>${prompt.name}</h6>
                <div>
                    <button class="btn btn-sm btn-outline-primary" onclick="editPrompt(${prompt.id})">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deletePrompt(${prompt.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <p>${prompt.text.substring(0, 50)}${prompt.text.length > 50 ? '...' : ''}</p>
            <small class="text-muted">${prompt.description}</small>
            <div class="variant-pills">
                ${prompt.variants.map(variant => `
                    <span class="variant-pill ${variant.steganographic ? 'steganographic' : ''} ${variant.font === 'webdings' ? 'webdings' : ''} ${variant.color !== '#000000' ? 'color' : ''}">
                        ${variant.name}
                    </span>
                `).join('')}
            </div>
        `;
        
        container.insertBefore(card, addButton);
    });
}

// Variant functions
function resetVariants() {
    const container = document.getElementById('variantsContainer');
    container.innerHTML = `
        <div class="style-editor" data-variant-id="1">
            <div class="row">
                <div class="col-md-3">
                    <label class="form-label">Variant Name</label>
                    <input type="text" class="form-control" value="A" data-field="name">
                </div>
                <div class="col-md-3">
                    <label class="form-label">Font</label>
                    <select class="form-select" data-field="font">
                        <option value="regular">Regular</option>
                        <option value="webdings">Webdings</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Size</label>
                    <input type="number" class="form-control" value="12" data-field="size">
                </div>
                <div class="col-md-2">
                    <label class="form-label">Style</label>
                    <select class="form-select" data-field="style">
                        <option value="regular">Regular</option>
                        <option value="steganographic">Hidden</option>
                        <option value="large_size">Large Size</option>
                        <option value="small_size">Small Size</option>
                        <option value="color_red">Red</option>
                        <option value="color_blue">Blue</option>
                        <option value="color_green">Green</option>
                        <option value="color_light_grey">Light Grey</option>
                        <option value="color_white">White</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <div class="form-check mt-4">
                        <input class="form-check-input" type="checkbox" data-field="steganographic">
                        <label class="form-check-label">Hidden</label>
                    </div>
                </div>
            </div>
        </div>
    `;
    variantCounter = 1;
}

function loadVariants(variants) {
    const container = document.getElementById('variantsContainer');
    container.innerHTML = '';
    
    variants.forEach((variant, index) => {
        const variantDiv = document.createElement('div');
        variantDiv.className = 'style-editor';
        variantDiv.setAttribute('data-variant-id', index + 1);
        variantDiv.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <label class="form-label">Variant Name</label>
                    <input type="text" class="form-control" value="${variant.name}" data-field="name">
                </div>
                <div class="col-md-3">
                    <label class="form-label">Font</label>
                    <select class="form-select" data-field="font">
                        <option value="regular" ${variant.font === 'regular' ? 'selected' : ''}>Regular</option>
                        <option value="webdings" ${variant.font === 'webdings' ? 'selected' : ''}>Webdings</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Size</label>
                    <input type="number" class="form-control" value="${variant.size}" data-field="size">
                </div>
                <div class="col-md-2">
                    <label class="form-label">Style</label>
                    <select class="form-select" data-field="style">
                        <option value="regular" ${variant.style === 'regular' ? 'selected' : ''}>Regular</option>
                        <option value="steganographic" ${variant.style === 'steganographic' ? 'selected' : ''}>Hidden</option>
                        <option value="large_size" ${variant.style === 'large_size' ? 'selected' : ''}>Large Size</option>
                        <option value="small_size" ${variant.style === 'small_size' ? 'selected' : ''}>Small Size</option>
                        <option value="color_red" ${variant.style === 'color_red' ? 'selected' : ''}>Red</option>
                        <option value="color_blue" ${variant.style === 'color_blue' ? 'selected' : ''}>Blue</option>
                        <option value="color_green" ${variant.style === 'color_green' ? 'selected' : ''}>Green</option>
                        <option value="color_light_grey" ${variant.style === 'color_light_grey' ? 'selected' : ''}>Light Grey</option>
                        <option value="color_white" ${variant.style === 'color_white' ? 'selected' : ''}>White</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <div class="form-check mt-4">
                        <input class="form-check-input" type="checkbox" data-field="steganographic" ${variant.steganographic ? 'checked' : ''}>
                        <label class="form-check-label">Hidden</label>
                    </div>
                </div>
            </div>
        `;
        
        container.appendChild(variantDiv);
    });
    
    variantCounter = variants.length;
}

function addVariant() {
    const container = document.getElementById('variantsContainer');
    const variantDiv = document.createElement('div');
    variantDiv.className = 'style-editor';
    variantDiv.setAttribute('data-variant-id', ++variantCounter);
    
    // Generate next letter (A, B, C, etc.)
    const nextLetter = String.fromCharCode(65 + variantCounter - 1);
    
    variantDiv.innerHTML = `
        <div class="row">
            <div class="col-md-3">
                <label class="form-label">Variant Name</label>
                <input type="text" class="form-control" value="${nextLetter}" data-field="name">
            </div>
            <div class="col-md-3">
                <label class="form-label">Font</label>
                <select class="form-select" data-field="font">
                    <option value="regular">Regular</option>
                    <option value="webdings">Webdings</option>
                </select>
            </div>
            <div class="col-md-2">
                <label class="form-label">Size</label>
                <input type="number" class="form-control" value="12" data-field="size">
            </div>
            <div class="col-md-2">
                <label class="form-label">Style</label>
                <select class="form-select" data-field="style">
                    <option value="regular">Regular</option>
                    <option value="steganographic">Hidden</option>
                    <option value="large_size">Large Size</option>
                    <option value="small_size">Small Size</option>
                    <option value="color_red">Red</option>
                    <option value="color_blue">Blue</option>
                    <option value="color_green">Green</option>
                    <option value="color_light_grey">Light Grey</option>
                    <option value="color_white">White</option>
                </select>
            </div>
            <div class="col-md-2">
                <div class="form-check mt-4">
                    <input class="form-check-input" type="checkbox" data-field="steganographic">
                    <label class="form-check-label">Hidden</label>
                </div>
            </div>
        </div>
    `;
    
    container.appendChild(variantDiv);
}

// Body functions
function addBody() {
    currentEditingBody = null;
    document.getElementById('bodyName').value = '';
    document.getElementById('bodyText').value = '';
    document.getElementById('bodyFile').value = '';
    
    // Reset to text content
    document.getElementById('textContent').checked = true;
    document.getElementById('textContentSection').style.display = 'block';
    document.getElementById('fileContentSection').style.display = 'none';
    
    const modal = new bootstrap.Modal(document.getElementById('bodyModal'));
    modal.show();
}

function editBody(id) {
    const body = bodies.find(b => b.id === id);
    if (!body) return;
    
    currentEditingBody = id;
    document.getElementById('bodyName').value = body.name;
    
    if (body.type === 'text') {
        document.getElementById('textContent').checked = true;
        document.getElementById('bodyText').value = body.content;
        document.getElementById('textContentSection').style.display = 'block';
        document.getElementById('fileContentSection').style.display = 'none';
    } else {
        document.getElementById('fileContent').checked = true;
        document.getElementById('textContentSection').style.display = 'none';
        document.getElementById('fileContentSection').style.display = 'block';
    }
    
    const modal = new bootstrap.Modal(document.getElementById('bodyModal'));
    modal.show();
}

function saveBody() {
    const name = document.getElementById('bodyName').value.trim();
    const isTextContent = document.getElementById('textContent').checked;
    
    if (!name) {
        alert('Please enter a body name');
        return;
    }
    
    if (isTextContent) {
        const content = document.getElementById('bodyText').value.trim();
        if (!content) {
            alert('Please enter text content');
            return;
        }
        
        const bodyData = {
            id: currentEditingBody || ++bodyCounter,
            name: name,
            type: 'text',
            content: content
        };
        
        saveBodyData(bodyData);
    } else {
        const fileInput = document.getElementById('bodyFile');
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select a file');
            return;
        }
        
        const file = fileInput.files[0];
        
        // Upload file first
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const bodyData = {
                    id: currentEditingBody || ++bodyCounter,
                    name: name,
                    type: 'file',
                    file_path: data.filepath,
                    content: `[File: ${file.name}]`
                };
                
                saveBodyData(bodyData);
            } else {
                alert('File upload failed: ' + data.error);
            }
        })
        .catch(error => {
            alert('Error uploading file: ' + error);
        });
    }
}

function saveBodyData(bodyData) {
    if (currentEditingBody) {
        const index = bodies.findIndex(b => b.id === currentEditingBody);
        bodies[index] = bodyData;
    } else {
        bodies.push(bodyData);
    }
    
    renderBodies();
    updateCombinationCount();
    
    const modal = bootstrap.Modal.getInstance(document.getElementById('bodyModal'));
    modal.hide();
}

function deleteBody(id) {
    if (confirm('Are you sure you want to delete this body?')) {
        bodies = bodies.filter(b => b.id !== id);
        renderBodies();
        updateCombinationCount();
    }
}

function renderBodies() {
    const container = document.getElementById('bodiesContainer');
    const addButton = container.querySelector('.add-item-btn');
    
    // Clear existing body cards
    container.querySelectorAll('.item-card').forEach(card => card.remove());
    
    bodies.forEach(body => {
        const card = document.createElement('div');
        card.className = 'item-card';
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <h6>${body.name}</h6>
                <div>
                    <button class="btn btn-sm btn-outline-primary" onclick="editBody(${body.id})">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteBody(${body.id})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <p>${body.content.substring(0, 50)}${body.content.length > 50 ? '...' : ''}</p>
            <small class="text-muted">Type: ${body.type}</small>
        `;
        
        container.insertBefore(card, addButton);
    });
}

// Combination counting
function updateCombinationCount() {
    const totalVariants = prompts.reduce((sum, prompt) => sum + prompt.variants.length, 0);
    const totalCombinations = totalVariants * bodies.length * selectedFormats.length;
    
    document.getElementById('combinationCount').textContent = totalCombinations;
}

// Preview combinations
function previewCombinations() {
    if (prompts.length === 0 || bodies.length === 0 || selectedFormats.length === 0) {
        alert('Please add at least one prompt, one body, and select at least one format');
        return;
    }
    
    const combinations = [];
    
    prompts.forEach(prompt => {
        prompt.variants.forEach(variant => {
            bodies.forEach(body => {
                selectedFormats.forEach(format => {
                    const filename = `${prompt.name.replace(/\\s+/g, '_')}_${variant.name}_${body.name.replace(/\\s+/g, '_')}.${format}`;
                    combinations.push({
                        prompt: prompt.name,
                        variant: variant.name,
                        body: body.name,
                        format: format,
                        filename: filename
                    });
                });
            });
        });
    });
    
    let previewText = `Preview of ${combinations.length} combinations:\\n\\n`;
    combinations.slice(0, 10).forEach(combo => {
        previewText += `${combo.filename}\\n`;
    });
    
    if (combinations.length > 10) {
        previewText += `\\n... and ${combinations.length - 10} more documents`;
    }
    
    alert(previewText);
}

// Generate combinations
function generateCombinations() {
    if (prompts.length === 0 || bodies.length === 0 || selectedFormats.length === 0) {
        alert('Please add at least one prompt, one body, and select at least one format');
        return;
    }
    
    const projectTitle = document.getElementById('projectTitle').value.trim() || 'LLM Evaluation Test';
    const outputDir = document.getElementById('outputDir').value.trim() || 'generated_documents';
    
    // Calculate total combinations
    const totalVariants = prompts.reduce((sum, prompt) => sum + prompt.variants.length, 0);
    const totalCombinations = totalVariants * bodies.length * selectedFormats.length;
    
    // Show progress section
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = `Starting generation of ${totalCombinations} documents...`;
    
    // Prepare configuration
    const config = {
        title: projectTitle,
        output_dir: outputDir,
        prompts: prompts.map(prompt => ({
            name: prompt.name,
            text: prompt.text,
            description: prompt.description,
            variants: prompt.variants.map(variant => ({
                name: variant.name,
                font: variant.font,
                size: variant.size,
                color: variant.color,
                steganographic: variant.steganographic
            }))
        })),
        bodies: bodies.map(body => ({
            name: body.name,
            type: body.type,
            content: body.content,
            file_path: body.file_path
        })),
        formats: selectedFormats
    };
    
    // Start generation with progress simulation
    simulateProgress(totalCombinations);
    
    // Send to backend
    fetch('/api/documents/combinatorial', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        // Complete progress
        document.getElementById('progressBar').style.width = '100%';
        document.getElementById('progressText').textContent = 'Generation complete!';
        
        setTimeout(() => {
            document.getElementById('progressSection').style.display = 'none';
            
            if (data.success) {
                displayResults(data.results);
            } else {
                alert('Error: ' + data.error);
            }
        }, 1000);
    })
    .catch(error => {
        document.getElementById('progressSection').style.display = 'none';
        alert('Error generating documents: ' + error);
    });
}

// Simulate progress for better user experience
function simulateProgress(totalCombinations) {
    let currentProgress = 0;
    const maxProgress = 90; // Don't go to 100% until actually complete
    const interval = Math.max(100, Math.min(1000, 10000 / totalCombinations));
    
    const progressInterval = setInterval(() => {
        if (currentProgress < maxProgress) {
            currentProgress += Math.random() * 10;
            if (currentProgress > maxProgress) currentProgress = maxProgress;
            
            document.getElementById('progressBar').style.width = currentProgress + '%';
            document.getElementById('progressText').textContent = 
                `Generating documents... ${Math.round(currentProgress)}% complete`;
        }
    }, interval);
    
    // Store interval ID to clear it later
    window.progressInterval = progressInterval;
    
    // Clear the interval after a reasonable time
    setTimeout(() => {
        clearInterval(progressInterval);
    }, 30000); // 30 seconds max
}

// Display results
function displayResults(results) {
    const resultsSection = document.getElementById('resultsSection');
    const resultsGrid = document.getElementById('resultsGrid');
    
    resultsGrid.innerHTML = '';
    
    results.forEach(result => {
        const resultDiv = document.createElement('div');
        resultDiv.className = `result-item ${result.success ? 'success' : 'error'}`;
        
        if (result.success) {
            resultDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <h6>${result.filename}</h6>
                    <a href="/api/documents/download/${result.filename}" class="btn btn-sm btn-primary">
                        <i class="fas fa-download"></i>
                    </a>
                </div>
                <p class="text-success mb-1">
                    <i class="fas fa-check-circle"></i> Generated successfully
                </p>
                <small class="text-muted">
                    ${result.prompt} (${result.variant}) → ${result.body} → ${result.format.toUpperCase()}
                </small>
            `;
        } else {
            resultDiv.innerHTML = `
                <h6>${result.filename}</h6>
                <p class="text-danger mb-1">
                    <i class="fas fa-exclamation-circle"></i> Generation failed
                </p>
                <small class="text-muted">${result.error}</small>
            `;
        }
        
        resultsGrid.appendChild(resultDiv);
    });
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Download all results as ZIP
function downloadAll() {
    fetch('/api/documents/download-zip', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Failed to create ZIP file');
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'generated_documents.zip';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        alert('Error downloading ZIP file: ' + error);
    });
}
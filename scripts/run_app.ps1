# Launch AgenticRLHF from the canonical E: workspace with E: caches.
$ErrorActionPreference = "Stop"
Set-Location "E:\Projects\AgenticRLHF-RLAIF"

$env:HF_HOME = "E:\ml-cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE = "E:\ml-cache\huggingface\hub"
$env:TRANSFORMERS_CACHE = "E:\ml-cache\huggingface\transformers"
$env:OLLAMA_MODELS = "E:\ollama-models\models"

Write-Host "Workspace: $(Get-Location)"
Write-Host "HF_HOME=$env:HF_HOME"
Write-Host "OLLAMA_MODELS=$env:OLLAMA_MODELS"
streamlit run app.py

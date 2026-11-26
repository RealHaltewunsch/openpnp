# Start the OpenPnP vision service on Windows x64.
# Requires python3 with onnxruntime + Pillow + numpy installed (CPU or DirectML).
# NOTE: not yet verified on Windows (macOS is the development target for now).
param(
    [string]$ModelsDir = "models/synthetic",
    [int]$Port = 8010,
    [string]$Provider = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$args = @("-m", "vision_service", "--models-dir", $ModelsDir, "--host", "127.0.0.1", "--port", "$Port")
if ($Provider -ne "") {
    $args += @("--provider", $Provider)
}

& python -m vision_service @args

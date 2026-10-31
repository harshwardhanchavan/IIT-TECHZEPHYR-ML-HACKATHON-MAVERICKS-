# PowerShell launcher. If scripts are blocked run once:  powershell -ExecutionPolicy Bypass -File run.ps1 setup
param(
    [Parameter(Position = 0)][string]$Command = "app",
    [Parameter(ValueFromRemainingArguments = $true)]$Rest
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" }
}
function Use-Venv {
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Host "[!] Environment not found. Run:  .\run.ps1 setup" -ForegroundColor Red
        exit 1
    }
    . .\.venv\Scripts\Activate.ps1
}

switch ($Command) {
    "setup" {
        $useLauncher = $false
        if (Get-Command py -ErrorAction SilentlyContinue) {
            py -3.10 --version 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { $useLauncher = $true }
        }
        if ($useLauncher) { Run py -3.10 -m venv .venv } else { Run python -m venv .venv }
        . .\.venv\Scripts\Activate.ps1
        Run python -m pip install --upgrade pip wheel setuptools
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            Run pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
        } else {
            Write-Host "[warn] No NVIDIA GPU detected - installing CPU PyTorch. Generation needs a GPU or Google Colab." -ForegroundColor Yellow
            Run pip install torch==2.3.1 torchvision==0.18.1
        }
        Run pip install -r requirements.txt
        if (-not (Test-Path "models\TripoSR")) { Run git clone https://github.com/VAST-AI-Research/TripoSR.git models\TripoSR }
        Run python create_examples.py
        Write-Host "`nSetup finished. Next:  .\run.ps1 warmup   (downloads ~10 GB of weights)   then   .\run.ps1 app"
    }
    "app"      { Use-Venv; python app.py @Rest }
    "cli"      { Use-Venv; python cli.py @Rest }
    "examples" { Use-Venv; python create_examples.py }
    "warmup"   { Use-Venv; python cli.py --warmup }
    "train"    { Use-Venv; python train.py @Rest }
    "test"     { Use-Venv; python -m pytest -q tests }
    default {
        Write-Host @"

  .\run.ps1 setup      create .venv, install PyTorch + dependencies, clone TripoSR
  .\run.ps1 warmup     download and load all model weights once
  .\run.ps1 app        start the web app (add --share for a public link)
  .\run.ps1 cli ...    e.g.  .\run.ps1 cli examples\mug.png --prompt "red mug"
  .\run.ps1 examples   regenerate example sketches
  .\run.ps1 train      train the optional QuickDraw classifier (add --quick)
  .\run.ps1 test       run unit tests
"@
    }
}

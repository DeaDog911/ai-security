# Создание виртуального окружения проекта ai-security (Windows, PowerShell).
#
# Запуск из корня проекта:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1           # PyTorch с ROCm (AMD GPU)
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Cpu      # PyTorch только для CPU
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Recreate # пересоздать venv
#
# Окружение создаётся командой:  python -m virtualenv --python python312 venv
# Python 3.12 должен быть установлен (колёса ROCm от AMD собраны только под cp312).
# Требования для ROCm: Windows 11, драйвер AMD Adrenalin 26.2.2 или новее.
# Источник колёс: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html

param(
    [switch]$Cpu,
    [switch]$Recreate,
    [string]$PythonSpec = "python312"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Venv = Join-Path $ProjectRoot "venv"
$Py = Join-Path $Venv "Scripts\python.exe"

$RocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
$RocmSdk = @(
    "$RocmBase/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "$RocmBase/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "$RocmBase/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "$RocmBase/rocm-7.2.1.tar.gz"
)
$RocmTorch = "$RocmBase/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Run([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Команда завершилась с кодом ${LASTEXITCODE}: $exe $($argv -join ' ')" }
}

# 1. Виртуальное окружение через virtualenv
if ($Recreate -and (Test-Path $Venv)) {
    Step "Удаление старого venv"
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path $Py)) {
    Step "Проверка virtualenv"
    & python -m virtualenv --version
    if ($LASTEXITCODE -ne 0) {
        Write-Host "virtualenv не найден, устанавливаю: python -m pip install --user virtualenv"
        Run "python" @("-m", "pip", "install", "--user", "virtualenv")
    }
    Step "Создание venv: python -m virtualenv --python $PythonSpec venv"
    & python -m virtualenv --python $PythonSpec $Venv
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось создать venv на $PythonSpec. Убедитесь, что Python 3.12 установлен (winget install Python.Python.3.12), и откройте новое окно PowerShell."
    }
} else {
    Write-Host "venv уже существует, использую его (для пересоздания: -Recreate)"
}
$ver = & $Py -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "Python в venv: $ver"
if (-not $Cpu -and $ver -ne "3.12") {
    throw "Для ROCm нужен Python 3.12, а в venv $ver. Пересоздайте: -Recreate (или используйте -Cpu)."
}
Run $Py @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

# 2. PyTorch
if ($Cpu) {
    Step "Установка PyTorch (CPU)"
    Run $Py @("-m", "pip", "install", "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cpu")
} else {
    Step "Установка ROCm SDK для Windows (может занять несколько минут)"
    Run $Py (@("-m", "pip", "install", "--no-cache-dir") + $RocmSdk)
    Step "Установка PyTorch 2.9.1 + ROCm 7.2.1"
    Run $Py @("-m", "pip", "install", "--no-cache-dir", $RocmTorch)
}

# 3. Остальные зависимости
Step "Установка зависимостей из requirements.txt"
Run $Py @("-m", "pip", "install", "-r", (Join-Path $ProjectRoot "requirements.txt"))

# 4. Ядро Jupyter
Step "Регистрация ядра Jupyter"
Run $Py @("-m", "ipykernel", "install", "--user", "--name", "ai-security", "--display-name", "Python (ai-security)")

# 5. Проверка
Step "Проверка устройства"
& $Py (Join-Path $ProjectRoot "src\check_device.py")

Write-Host "`nГотово. Активация окружения: .\venv\Scripts\Activate.ps1" -ForegroundColor Green

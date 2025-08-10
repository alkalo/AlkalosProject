# install_env.ps1
# Instala dependencias del proyecto evitando compilaciones

Write-Host "==> Actualizando pip, setuptools, wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "==> Instalando paquetes grandes (wheels)..." -ForegroundColor Cyan
pip install --only-binary=:all: --index-url https://pypi.org/simple `
    numpy==2.1.3 `
    scipy==1.14.1 `
    scikit-learn==1.6.1 `
    pandas==2.3.1 `
    matplotlib==3.9.2

Write-Host "==> Instalando resto de requirements..." -ForegroundColor Cyan
pip install -r requirements.txt --no-deps

Write-Host "✅ Entorno listo. Puedes ejecutar .\sanity_e2e.ps1" -ForegroundColor Green

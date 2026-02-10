# SageMaker Setup for MeltingPot MARL Training

Ten folder zawiera wszystkie pliki potrzebne do uruchomienia eksperymentów na AWS SageMaker.

## Szybki start

### 1. Wymagania wstępne

```bash
# Zainstaluj AWS CLI i skonfiguruj credentials
pip install awscli boto3 sagemaker tabulate
aws configure
```

### 2. Ustaw zmienne środowiskowe

```bash
export WANDB_API_KEY="twój-klucz-wandb"
export AWS_DEFAULT_REGION="eu-central-1"
```

### 3. Zbuduj i wypchnij obraz Docker

```bash
# Pierwszy raz - zbuduj obraz i wypchnij do ECR
python sagemaker_scripts/launch_training.py --build-image --exp private --seed 123

# Lub tylko zbuduj obraz (dry-run nie uruchomi treningu, ale zbuduje obraz)
python sagemaker_scripts/launch_training.py --build-image --dry-run --exp private
```

### 4. Uruchom pojedynczy eksperyment

```bash
# Eksperyment "private" z seed=123
python sagemaker_scripts/launch_training.py \
    --exp private \
    --seed 123 \
    --image-uri <twój-ecr-uri>
```

### 5. Uruchom wszystkie eksperymenty

```bash
# Uruchomi 10 eksperymentów (2 środowiska × 5 seedów)
python sagemaker_scripts/launch_training.py \
    --run-all \
    --image-uri <twój-ecr-uri> \
    --budget 1000
```

## Monitorowanie

```bash
# Sprawdź status wszystkich jobów
python sagemaker_scripts/monitor_jobs.py --status --region eu-central-1

# Pobierz wyniki ukończonych eksperymentów
python sagemaker_scripts/monitor_jobs.py --download --output-dir ./results_sagemaker --region eu-central-1

# Zatrzymaj wszystkie joby (awaryjnie)
python sagemaker_scripts/monitor_jobs.py --stop-all --region eu-central-1
```

## Struktura plików

```
sagemaker_scripts/
├── Dockerfile              # Obraz Docker dla SageMaker
├── train_sagemaker.py      # Skrypt treningowy (entry point)
├── launch_training.py      # Launcher eksperymentów
├── monitor_jobs.py         # Monitorowanie i pobieranie wyników
├── requirements-docker.txt # Zależności dla obrazu Docker
├── requirements-sagemaker.txt # Zależności lokalne (boto3, sagemaker)
└── README.md               # Ta dokumentacja
```

## Koszty

| Typ instancji | GPU | vCPU | RAM | Cena/h | Rekomendacja |
|---------------|-----|------|-----|--------|--------------|
| ml.g4dn.xlarge | 1× T4 | 4 | 16GB | $0.526 | ✅ Najlepsza wartość |
| ml.g4dn.2xlarge | 1× T4 | 8 | 32GB | $0.752 | Więcej RAM |
| ml.g5.xlarge | 1× A10G | 4 | 16GB | $1.006 | Szybsze GPU |
| ml.p3.2xlarge | 1× V100 | 8 | 61GB | $3.06 | Najszybsze |

### Szacunkowe koszty dla pełnego eksperymentu

- **10 eksperymentów** (2 środowiska × 5 seedów)
- **~15h na eksperyment** (10k iteracji)
- **ml.g4dn.xlarge**: 10 × 15h × $0.526 = **~$79**

Z budżetem $1000 możesz uruchomić:
- ~190 eksperymentów na ml.g4dn.xlarge
- Lub eksperymenty z większą liczbą iteracji
- Lub użyć szybszych instancji

## Integracja z WandB

Wyniki są automatycznie logowane do WandB jeśli ustawisz `WANDB_API_KEY`.

Dashboard: https://wandb.ai/twój-username/meltingpot-sagemaker

## Troubleshooting

### "No SageMaker execution role"
```bash
# Podaj ARN roli ręcznie
python sagemaker_scripts/launch_training.py --role arn:aws:iam::123456789012:role/SageMakerRole ...
```

### "Docker build failed"
```bash
# Upewnij się że Docker działa
docker info

# Zaloguj się do ECR
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-central-1.amazonaws.com
```

### "Job failed - ResourceLimitExceeded"
```bash
# Poproś o zwiększenie limitów w AWS Console
# Service Quotas → SageMaker → ml.g4dn.xlarge for training job usage
```

## Porównanie z oryginalnym setupem

| Aspekt | Lokalnie/EC2 | SageMaker |
|--------|--------------|-----------|
| Setup | Ręczny | Automatyczny |
| Skalowanie | Trudne | Łatwe |
| Koszty | Płacisz za całość | Płacisz za użycie |
| Checkpointy | Lokalne | S3 (trwałe) |
| Monitorowanie | Ręczne | Wbudowane + WandB |
| Równoległość | Ograniczona | Nieograniczona |

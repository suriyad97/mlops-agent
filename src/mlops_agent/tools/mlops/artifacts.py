"""
Artifact Generation Tools
===========================
Generate Dockerfile, Azure ML pipeline YAMLs, and AzDO CI YAML
based on the scanned project understanding.
All generated files are committed back to the Azure Repo.
"""

from langchain_core.tools import tool


@tool
def generate_dockerfile(
    base_image: str,
    framework: str,
    requirements_content: str,
    python_version: str = "3.10"
) -> str:
    """
    Generates a Dockerfile for the ML project based on the detected framework.

    Args:
        base_image: Docker base image (e.g. "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04")
        framework: Detected ML framework (e.g. "xgboost", "pytorch", "sklearn")
        requirements_content: Full content of requirements.txt
        python_version: Python version to use (default: "3.10")

    Returns:
        Dockerfile content as a string (commit this to the repo)
    """
    packages = "\n".join(
        [f"RUN pip install {line.strip()}"
         for line in requirements_content.splitlines()
         if line.strip() and not line.startswith("#")]
    )

    return f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install ML framework and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "--version"]
"""


@tool
def generate_aml_environment_yaml(
    environment_name: str,
    acr_image_tag: str,
    acr_registry: str
) -> str:
    """
    Generates an Azure ML Environment YAML that pulls the Docker image from ACR.

    Args:
        environment_name: Name to register the environment as in AML
        acr_image_tag: The Docker image tag in ACR (e.g. "mlops-env:latest")
        acr_registry: ACR login server (e.g. "myregistry.azurecr.io")

    Returns:
        AML Environment YAML content as a string
    """
    return f"""$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json

name: {environment_name}
version: 1
description: ML training and inference environment built from source

image: {acr_registry}/{acr_image_tag}

inference_config:
  liveness_route:
    path: /
    port: 5001
  readiness_route:
    path: /
    port: 5001
  scoring_route:
    path: /score
    port: 5001
"""


@tool
def generate_ct_pipeline_yaml(
    environment_name: str,
    compute_cluster: str,
    train_script: str,
    model_name: str,
    task_type: str,
    primary_metric: str
) -> str:
    """
    Generates the Continuous Training (CT) pipeline YAML for Azure ML.

    Args:
        environment_name: AML environment to use for training
        compute_cluster: AML compute cluster name
        train_script: Path to training script (e.g. "train.py")
        model_name: Name to register the model under in AML registry
        task_type: "classification", "regression", "multiclass", "timeseries"
        primary_metric: Metric to track (e.g. "auc", "f1", "rmse")

    Returns:
        CT Pipeline YAML content as a string
    """
    return f"""$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json

type: pipeline
display_name: Continuous Training Pipeline
description: Train model on new data, register as challenger

inputs:
  data_path:
    type: uri_folder

outputs:
  model_output:
    type: uri_folder
    mode: rw_mount

settings:
  default_compute: azureml:{compute_cluster}

jobs:
  data_validation:
    type: command
    code: ./src
    command: python validate_data.py --data_path ${{{{inputs.data_path}}}}
    environment: azureml:{environment_name}@latest

  training:
    type: command
    code: ./src
    command: >-
      python {train_script}
      --data_path ${{{{inputs.data_path}}}}
      --model_output ${{{{outputs.model_output}}}}
      --task_type {task_type}
      --primary_metric {primary_metric}
    environment: azureml:{environment_name}@latest
    inputs:
      data_path: ${{{{inputs.data_path}}}}
    outputs:
      model_output: ${{{{outputs.model_output}}}}
    depends_on:
      - data_validation

  register_model:
    type: command
    code: ./src
    command: >-
      python register_model.py
      --model_path ${{{{inputs.model_output}}}}
      --model_name {model_name}
      --label challenger
      --metric {primary_metric}
    environment: azureml:{environment_name}@latest
    inputs:
      model_output: ${{{{jobs.training.outputs.model_output}}}}
    depends_on:
      - training
"""


@tool
def generate_inference_pipeline_yaml(
    environment_name: str,
    compute_cluster: str,
    score_script: str,
    model_name: str
) -> str:
    """
    Generates the batch inference pipeline YAML for Azure ML.

    Args:
        environment_name: AML environment to use for inference
        compute_cluster: AML compute cluster name
        score_script: Path to inference/scoring script (e.g. "score.py")
        model_name: Name of the champion model to load from registry

    Returns:
        Inference Pipeline YAML content as a string
    """
    return f"""$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json

type: pipeline
display_name: Batch Inference Pipeline
description: Run batch scoring using the champion model on unseen data

inputs:
  input_data:
    type: uri_folder

outputs:
  predictions:
    type: uri_folder
    mode: rw_mount

settings:
  default_compute: azureml:{compute_cluster}

jobs:
  scoring:
    type: command
    code: ./src
    command: >-
      python {score_script}
      --input_path ${{{{inputs.input_data}}}}
      --output_path ${{{{outputs.predictions}}}}
      --model_name {model_name}
      --model_label champion
    environment: azureml:{environment_name}@latest
    inputs:
      input_data: ${{{{inputs.input_data}}}}
    outputs:
      predictions: ${{{{outputs.predictions}}}}
"""


@tool
def generate_drift_pipeline_yaml(
    environment_name: str,
    compute_cluster: str,
    task_type: str,
    drift_type: str
) -> str:
    """
    Generates a drift monitoring pipeline YAML.

    Args:
        environment_name: AML environment name
        compute_cluster: AML compute cluster name
        task_type: "classification" or "regression" (affects drift method)
        drift_type: "data_drift", "prediction_drift", or "concept_drift"

    Returns:
        Drift Pipeline YAML content as a string
    """
    script_map = {
        "data_drift": "monitor_data_drift.py",
        "prediction_drift": "monitor_prediction_drift.py",
        "concept_drift": "monitor_concept_drift.py",
    }
    script = script_map.get(drift_type, "monitor_data_drift.py")

    return f"""$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json

type: pipeline
display_name: {drift_type.replace("_", " ").title()} Pipeline
description: Monitor {drift_type.replace("_", " ")} in production

inputs:
  reference_data:
    type: uri_folder
  current_data:
    type: uri_folder

outputs:
  drift_report:
    type: uri_folder
    mode: rw_mount

settings:
  default_compute: azureml:{compute_cluster}

jobs:
  drift_detection:
    type: command
    code: ./src
    command: >-
      python {script}
      --reference_data ${{{{inputs.reference_data}}}}
      --current_data ${{{{inputs.current_data}}}}
      --output_path ${{{{outputs.drift_report}}}}
      --task_type {task_type}
    environment: azureml:{environment_name}@latest
    inputs:
      reference_data: ${{{{inputs.reference_data}}}}
      current_data: ${{{{inputs.current_data}}}}
    outputs:
      drift_report: ${{{{outputs.drift_report}}}}
"""


@tool
def generate_azdo_ci_yaml(
    acr_registry: str,
    image_name: str,
    aml_environment_name: str,
    service_connection: str = "azure-service-connection"
) -> str:
    """
    Generates the Azure DevOps CI pipeline YAML.
    Builds the Docker image, pushes to ACR, and registers the AML environment.

    Args:
        acr_registry: ACR login server (e.g. "myregistry.azurecr.io")
        image_name: Docker image name (e.g. "mlops-env")
        aml_environment_name: AML environment name to register after build
        service_connection: AzDO service connection name for Azure auth

    Returns:
        AzDO CI pipeline YAML content as a string
    """
    return f"""trigger:
  branches:
    include:
      - main
  paths:
    include:
      - Dockerfile
      - requirements.txt
      - src/**

pool:
  vmImage: ubuntu-latest

variables:
  acrRegistry: '{acr_registry}'
  imageName: '{image_name}'
  imageTag: $(Build.BuildId)
  amlEnvironmentName: '{aml_environment_name}'

stages:
  - stage: BuildAndPush
    displayName: Build Docker Image and Push to ACR
    jobs:
      - job: Build
        steps:
          - task: AzureCLI@2
            displayName: Build and Push Docker Image
            inputs:
              azureSubscription: '{service_connection}'
              scriptType: bash
              scriptLocation: inlineScript
              inlineScript: |
                az acr login --name $(acrRegistry)
                docker build -t $(acrRegistry)/$(imageName):$(imageTag) .
                docker build -t $(acrRegistry)/$(imageName):latest .
                docker push $(acrRegistry)/$(imageName):$(imageTag)
                docker push $(acrRegistry)/$(imageName):latest
                echo "Image pushed: $(acrRegistry)/$(imageName):$(imageTag)"

  - stage: RegisterAMLEnvironment
    displayName: Register Azure ML Environment
    dependsOn: BuildAndPush
    jobs:
      - job: Register
        steps:
          - task: AzureCLI@2
            displayName: Register AML Environment
            inputs:
              azureSubscription: '{service_connection}'
              scriptType: bash
              scriptLocation: inlineScript
              inlineScript: |
                az ml environment create \\
                  --file environments/aml_environment.yml \\
                  --set version=$(Build.BuildId) \\
                  --workspace-name $AML_WORKSPACE_NAME \\
                  --resource-group $AML_RESOURCE_GROUP
                echo "AML Environment registered: $(amlEnvironmentName):$(Build.BuildId)"
"""

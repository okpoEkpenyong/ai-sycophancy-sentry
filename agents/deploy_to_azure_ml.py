from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

#credential=DefaultAzureCredential()

# With this:
from azure.identity import InteractiveBrowserCredential
credential = InteractiveBrowserCredential(tenant_id="1554c886-6d7c-464b-acff-af4070546677")

ml_client = MLClient(
    credential=credential,
    subscription_id="d96199a8-52d6-47d2-9e24-dd38c47c3791",
    resource_group_name="okpo.ekpenyong-rg",
    workspace_name= "gpu-ssus-ws",
)

endpoint_name = "sycophancy-sentry-prober"

# 2. Create the Endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Mechanistic Probing Backend for Sycophancy-Sentry",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 3. Create the Deployment
deployment = ManagedOnlineDeployment(
    name="qwen-prober-v1",
    endpoint_name=endpoint_name,
    code_configuration=CodeConfiguration(
        code="./", 
        scoring_script="score.py" # Point to the Azure-friendly script
    ),
    environment=Environment(
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest",
    ),
    instance_type="Standard_NC16as_T4_v3", # Good choice for 8B model
    instance_count=1,
)

# Start the deployment (this takes 10-20 minutes)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# 4. SET TRAFFIC TO 100%
endpoint.traffic = {"qwen-prober-v1": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Add this at the very end of your deploy_to_azure_ml.py script inside a try/except
try:
    ml_client.online_deployments.begin_create_or_update(deployment).result()
except Exception as e:
    print(f"Detailed Error: {e}")

# 5. RETRIEVE THE VALID URL AND KEY FOR MARKETPLACE SAAS
final_endpoint = ml_client.online_endpoints.get(name=endpoint_name)
keys = ml_client.online_endpoints.get_keys(name=endpoint_name)

print(f"\n" + "="*50)
print(f"🚀 SENTRY PROBER LIVE AT:")
print(f"URL: {final_endpoint.scoring_uri}")
print(f"KEY: {keys.primary_key}")
print("="*50)
